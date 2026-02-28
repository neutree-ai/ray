import asyncio
import logging
import os
import tempfile
from typing import List, Optional

from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin

default_logger = logging.getLogger(__name__)

_CONTAINER_RUNTIME_ENV_VAR = "RAY_EXPERIMENTAL_RUNTIME_ENV_CONTAINER_RUNTIME"


def _get_container_runtime() -> str:
    """Get the container runtime to use.

    Returns "docker" or "podman" based on the
    RAY_EXPERIMENTAL_RUNTIME_ENV_CONTAINER_RUNTIME environment variable.
    Defaults to "podman" for backward compatibility.
    """
    runtime = os.environ.get(_CONTAINER_RUNTIME_ENV_VAR, "podman").lower()
    if runtime not in ("docker", "podman"):
        raise ValueError(
            f"Invalid {_CONTAINER_RUNTIME_ENV_VAR} value: '{runtime}'. "
            "Must be 'docker' or 'podman'."
        )
    return runtime


async def _create_impl(image_uri: str, logger: logging.Logger):
    # Pull image if it doesn't exist
    # Also get path to `default_worker.py` inside the image.
    container_runtime = _get_container_runtime()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chmod(tmpdir, 0o777)
        result_file = os.path.join(tmpdir, "worker_path.txt")
        get_worker_path_script = """
import ray._private.workers.default_worker as dw
with open('/shared/worker_path.txt', 'w') as f:
    f.write(dw.__file__)
"""
        # Use `:Z` volume flag only for podman (SELinux rootless support)
        volume_mount = (
            f"{tmpdir}:/shared:Z"
            if container_runtime == "podman"
            else f"{tmpdir}:/shared"
        )
        cmd = [
            container_runtime,
            "run",
            "--rm",
            "-v",
            volume_mount,
            image_uri,
            "python",
            "-c",
            get_worker_path_script,
        ]

        logger.info("Pulling image %s", image_uri)

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(
                f"{container_runtime} command failed: cmd={cmd}, returncode={process.returncode}, stdout={stdout.decode()}, stderr={stderr.decode()}"
            )

        if not os.path.exists(result_file):
            raise FileNotFoundError(
                f"Worker path file not created when getting worker path for image {image_uri}"
            )

        with open(result_file, "r") as f:
            worker_path = f.read().strip()

        if not worker_path.endswith(".py"):
            raise ValueError(
                f"Invalid worker path inferred in image {image_uri}: {worker_path}"
            )

        logger.info(f"Inferred worker path in image {image_uri}: {worker_path}")
        return worker_path


def _modify_context_impl(
    image_uri: str,
    worker_path: str,
    run_options: Optional[List[str]],
    context: RuntimeEnvContext,
    logger: logging.Logger,
    ray_tmp_dir: str,
):
    context.override_worker_entrypoint = worker_path

    container_runtime = _get_container_runtime()
    container_command = [
        container_runtime,
        "run",
        "-v",
        ray_tmp_dir + ":" + ray_tmp_dir,
        "--network=host",
        "--pid=host",
        "--ipc=host",
    ]

    # Podman-specific flags for rootless container support.
    # --cgroup-manager=cgroupfs: use cgroupfs for cgroup management.
    # --userns=keep-id: maps the user as itself (instead of as root)
    # into the container, so that mounted volumes are accessible.
    # https://www.redhat.com/sysadmin/rootless-podman-user-namespace-modes
    if container_runtime == "podman":
        container_command.extend([
            "--cgroup-manager=cgroupfs",
            "--userns=keep-id",
        ])

    # Environment variables to set in container
    env_vars = dict()

    # Propagate all host environment variables that have the prefix "RAY_"
    # This should include RAY_RAYLET_PID
    for env_var_name, env_var_value in os.environ.items():
        if env_var_name.startswith("RAY_"):
            env_vars[env_var_name] = env_var_value

    # Support for runtime_env['env_vars']
    env_vars.update(context.env_vars)

    # Set environment variables
    for env_var_name, env_var_value in env_vars.items():
        container_command.append("--env")
        container_command.append(f"{env_var_name}='{env_var_value}'")

    # The RAY_JOB_ID environment variable is needed for the default worker.
    # It won't be set at the time setup() is called, but it will be set
    # when worker command is executed, so we use RAY_JOB_ID=$RAY_JOB_ID
    # for the container start command
    container_command.append("--env")
    container_command.append("RAY_JOB_ID=$RAY_JOB_ID")

    if run_options:
        container_command.extend(run_options)
    # TODO(chenk008): add resource limit
    container_command.append("--entrypoint")
    container_command.append("python")
    container_command.append(image_uri)

    # Example (podman):
    # podman run -v /tmp/ray:/tmp/ray --network=host --pid=host --ipc=host
    # --cgroup-manager=cgroupfs --userns=keep-id
    # --env RAY_RAYLET_PID=23478 --env RAY_JOB_ID=$RAY_JOB_ID
    # --entrypoint python rayproject/ray:nightly-py39
    # Example (docker):
    # docker run -v /tmp/ray:/tmp/ray --network=host --pid=host --ipc=host
    # --env RAY_RAYLET_PID=23478 --env RAY_JOB_ID=$RAY_JOB_ID
    # --entrypoint python rayproject/ray:nightly-py39
    container_command_str = " ".join(container_command)
    logger.info(f"Starting worker in container with prefix {container_command_str}")

    context.py_executable = container_command_str


class ImageURIPlugin(RuntimeEnvPlugin):
    """Starts worker in a container of a custom image."""

    name = "image_uri"

    @staticmethod
    def get_compatible_keys():
        return {"image_uri", "config", "env_vars"}

    def __init__(self, ray_tmp_dir: str):
        self._ray_tmp_dir = ray_tmp_dir

    async def create(
        self,
        uri: Optional[str],
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: RuntimeEnvContext,
        logger: logging.Logger,
    ) -> float:
        if not runtime_env.image_uri():
            return

        self.worker_path = await _create_impl(runtime_env.image_uri(), logger)

    def modify_context(
        self,
        uris: List[str],
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: RuntimeEnvContext,
        logger: Optional[logging.Logger] = default_logger,
    ):
        if not runtime_env.image_uri():
            return

        _modify_context_impl(
            runtime_env.image_uri(),
            self.worker_path,
            [],
            context,
            logger,
            self._ray_tmp_dir,
        )


class ContainerPlugin(RuntimeEnvPlugin):
    """Starts worker in container."""

    name = "container"

    def __init__(self, ray_tmp_dir: str):
        self._ray_tmp_dir = ray_tmp_dir

    async def create(
        self,
        uri: Optional[str],
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: RuntimeEnvContext,
        logger: logging.Logger,
    ) -> float:
        if not runtime_env.has_py_container() or not runtime_env.py_container_image():
            return

        self.worker_path = await _create_impl(runtime_env.py_container_image(), logger)

    def modify_context(
        self,
        uris: List[str],
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: RuntimeEnvContext,
        logger: Optional[logging.Logger] = default_logger,
    ):
        if not runtime_env.has_py_container() or not runtime_env.py_container_image():
            return

        if runtime_env.py_container_worker_path():
            logger.warning(
                "You are using `container.worker_path`, but the path to "
                "`default_worker.py` is now automatically detected from the image. "
                "`container.worker_path` is deprecated and will be removed in future "
                "versions."
            )

        _modify_context_impl(
            runtime_env.py_container_image(),
            runtime_env.py_container_worker_path() or self.worker_path,
            runtime_env.py_container_run_options(),
            context,
            logger,
            self._ray_tmp_dir,
        )
