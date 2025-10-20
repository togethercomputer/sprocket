#!/usr/bin/env python3
"""jig - Simple deployment tool for Together AI"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "rich"]
# ///

import os
import sys
import json
import shlex
import shutil
import argparse
import subprocess
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Optional, Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib
try:
    import requests
except ImportError:
    try:
        import pip._vendor.requests as requests
    except ImportError:
        print("ERROR: requests not available", file=sys.stderr)
        sys.exit(1)

try:
    from rich.pretty import pprint
except ImportError:
    try:
        from pip._vendor.rich.pretty import pprint
    except ImportError:

        def pprint(data, **kwargs):
            print(json.dumps(data, indent=2))

# --- Configuration ---

TOGETHER_ENV = os.getenv("TOGETHER_ENV", "prod")
if TOGETHER_ENV == "prod":
    API_URL = "api.together.ai"
    REGISTRY_URL = "registry.together.xyz"
elif TOGETHER_ENV == "qa":
    API_URL = "api.qa.together.ai"
    REGISTRY_URL = "registry.t6r-ai.dev"
elif TOGETHER_ENV == "local":
    if "TOGETHER_API_URL" not in os.environ or "TOGETHER_REGISTRY_URL" not in os.environ:
        print("ERROR: TOGETHER_API_URL and TOGETHER_REGISTRY_URL must be set for local env", file=sys.stderr)
        sys.exit(1)
    API_URL = os.environ["TOGETHER_API_URL"]
    REGISTRY_URL = os.environ["TOGETHER_REGISTRY_URL"]
else:
    print("ERROR: unknown together env", TOGETHER_ENV)
    sys.exit(1)

GENERATE_DOCKERFILE = os.getenv("GENERATE_DOCKERFILE", "0") != "0"
DEBUG = os.getenv("TOGETHER_DEBUG", "").strip()[:1] in ("y", "1", "t")
AUTOSCALING_PROFILES = {
    "QueueBacklogPerWorker": [
        "targetValue"
    ]
}


@dataclass
class ImageConfig:
    """Container image configuration from pyproject.toml"""

    python_version: str = "3.11"
    system_packages: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    run: list[str] = field(default_factory=list)
    cmd: str = "python app.py"
    copy: list[str] = field(default_factory=list)
    auto_include_git: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "ImageConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class DeployConfig:
    """Deployment configuration"""

    description: str = ""
    gpu_type: str = "h100-80gb"
    gpu_count: int = 1
    cpu: int = 1
    memory: int = 8
    min_replicas: int = 1
    max_replicas: int = 1
    port: int = 8000
    environment_variables: dict[str, str] = field(default_factory=dict)
    command: Optional[list[str]] = None
    autoscaling: dict[str, str] = field(default_factory=dict)
    health_check_path: str = "/health"

    @classmethod
    def from_dict(cls, data: dict) -> "DeployConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class Config:
    """Main configuration from pyproject.toml"""

    model_name: Optional[str] = None
    dockerfile: str = "Dockerfile"
    image: ImageConfig = field(default_factory=ImageConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)

    @classmethod
    def load(cls, path: Path = Path("pyproject.toml")) -> "Config":
        """Load configuration from pyproject.toml"""
        if not path.exists():
            return cls()

        with open(path, "rb") as f:
            data = tomllib.load(f)

        jig_config = data.get("tool", {}).get("jig", {})
        name = jig_config.get("name") or data.get("project", {}).get("name", "")
        if not name:
            name = Path.cwd().name
            print(f"\N{PACKAGE} Name not set in pyproject.toml - defaulting to {name}")

        if autoscaling := jig_config.get("autoscaling", {}):
            autoscaling_profile = autoscaling.get("profile", "")
            if autoscaling_profile not in AUTOSCALING_PROFILES:
                print(
                    f"ERROR: Specify one of the supported autoscaling profiles: {AUTOSCALING_PROFILES.keys()}"
                )
                sys.exit(1)
        
            for required_param in AUTOSCALING_PROFILES[autoscaling_profile]:
                if not autoscaling.get(required_param):
                    print(f"ERROR: Autoscaling profile '{autoscaling_profile}' requires '{required_param}' to be set")
                    sys.exit(1)

            autoscaling["model"] = name
            jig_config["deploy"]["autoscaling"] = autoscaling

        return cls(
            image=ImageConfig.from_dict(jig_config.get("image", {})),
            deploy=DeployConfig.from_dict(jig_config["deploy"]),
            dockerfile=jig_config.get("dockerfile", "Dockerfile"),
            model_name=name,
        )


# --- State Management ---


@dataclass
class State:
    """Persistent state"""

    username: Optional[str] = None
    secrets: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = Path(".jig.json")) -> "State":
        """Load state from file"""
        try:
            with open(path) as f:
                return cls(**json.load(f))
        except FileNotFoundError:
            return cls()

    def save(self, path: Path = Path(".jig.json")):
        """Save state to file"""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# --- API Client ---


class APIClient:
    """Together AI API client"""

    def __init__(self, api_key: str):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def request(self, method: str, endpoint: str, **kwargs) -> Optional[dict]:
        """Make API request with error handling"""
        url = f"https://{API_URL}{endpoint}"
        if DEBUG:
            print(method, url)
        response = requests.request(method, url, headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None

    def get_username(self) -> str:
        """Get username from proof-data endpoint"""
        data = self.request("GET", "/api/user/proof-data")
        assert data
        # Currently returns project ID as lowercase
        return data["projectId"].lower()


# --- Container Operations ---


def generate_dockerfile(config: Config) -> str:
    """Generate Dockerfile from config"""
    lines = []

    # Multi-stage build
    lines.append(f"FROM python:{config.image.python_version} AS builder")
    lines.append("")

    # System packages in builder
    sys_pkgs = " ".join(config.image.system_packages or [])
    if sys_pkgs:
        lines.append("RUN --mount=type=cache,target=/var/cache/apt \\")
        lines.append("  apt-get update && \\")
        lines.append("  DEBIAN_FRONTEND=noninteractive \\")
        lines.append(f"  apt-get install -y --no-install-recommends {sys_pkgs} && \\")
        lines.append("  apt-get clean && rm -rf /var/lib/apt/lists/*")
        lines.append("")

    # UV for package installation
    lines.append("COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv\n")

    # Install Python packages
    lines.append("WORKDIR /app")
    lines.append("COPY pyproject.toml .")
    lines.append("RUN --mount=type=cache,target=/root/.cache/uv \\")
    lines.append("  uv pip install --system --compile-bytecode .\n")

    # Final stage - slim image
    lines.append(f"FROM python:{config.image.python_version}-slim\n")

    # System packages in final image
    if sys_pkgs:
        lines.append("RUN --mount=type=cache,target=/var/cache/apt \\")
        lines.append("  apt-get update && \\")
        lines.append("  DEBIAN_FRONTEND=noninteractive \\")
        lines.append(f"  apt-get install -y --no-install-recommends {sys_pkgs} && \\")
        lines.append("  apt-get clean && rm -rf /var/lib/apt/lists/*\n")

    # Copy Python installation
    lines.append(
        f"COPY --from=builder /usr/local/lib/python{config.image.python_version} /usr/local/lib/python{config.image.python_version}"
    )
    lines.append("COPY --from=builder /usr/local/bin /usr/local/bin\n")

    # Tini for proper signal handling
    lines.append("COPY --from=krallin/ubuntu-tini:latest /usr/local/bin/tini /tini")
    lines.append('ENTRYPOINT ["/tini", "--"]\n')

    # Environment variables
    for key, value in config.image.environment.items():
        lines.append(f"ENV {key}={value}")
    if config.image.environment:
        lines.append("")

    # Run commands
    for cmd in config.image.run:
        lines.append(f"RUN {cmd}")
    if config.image.run:
        lines.append("")

    # Copy files (preserving directory structure)
    lines.append("WORKDIR /app")
    files_to_copy = get_files_to_copy(config)
    for file in files_to_copy:
        lines.append(f"COPY {file} {file}")
    lines.append(
        "RUN --mount=type=bind,source=.,target=/src cp /src/.worker.p* worker.py 2>/dev/null || true"
    )
    # this tag will set the X-Worker-Version header, used for rollout monitoring
    lines.append(
        "RUN --mount=type=bind,source=.,target=/src git -C /src describe --tags --exact-match > VERSION"
    )
    lines.append("")

    # CMD
    lines.append(f"CMD {json.dumps(shlex.split(config.image.cmd))}")

    return "\n".join(lines)


def get_files_to_copy(config: Config) -> list[str]:
    """Get list of files to copy"""
    files = set(config.image.copy)

    if config.image.auto_include_git:
        try:
            result = subprocess.run(
                ["git", "ls-files"], capture_output=True, text=True, check=True
            )
            # Check if repo is clean
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )
            if status.stdout.strip():
                raise RuntimeError(
                    "Git repository has uncommitted changes: auto_include_git not allowed."
                )

            git_files = result.stdout.strip().split("\n")
            files.update(f for f in git_files if f and f != ".")
        except subprocess.CalledProcessError:
            pass  # Not a git repo or git not available

    # Never allow bare "."
    if "." in files:
        raise ValueError("Copying '.' is not allowed. Please enumerate specific files.")

    return sorted(files)


# --- CLI Framework ---


class Arg:
    """Argument definition for CLI commands"""

    def __init__(
        self,
        name: str,
        type: type = str,
        default: Any = None,
        help: str = "",
        flag: bool = False,
    ):
        self.name = name
        self.type = type
        self.default = default
        self.help = help
        self.flag = flag


def arg(name: str, type: type = str, default: Any = None, help: str = ""):
    """Create an argument definition"""
    # Determine if this is a flag based on type and default
    flag = type is bool and default is False
    return Arg(name, type, default, help, flag)


def command(*args):
    """Decorator for CLI commands"""

    def decorator(func):
        # Store argument definitions
        func._cli_args = list(args)
        return func

    return decorator


class CLI:
    """Command line interface handler"""

    def __init__(self, app_class: type):
        self.app_class = app_class
        self.parser = argparse.ArgumentParser(
            description=app_class.__doc__ or "CLI Application"
        )
        self.subparsers = self.parser.add_subparsers(
            dest="command", help="Available commands"
        )

        # Find all command methods
        for name in dir(app_class):
            if name.startswith("_"):
                continue
            method = getattr(app_class, name)
            if hasattr(method, "_cli_args"):
                self._add_command(name, method)

    def _add_command(self, name: str, method):
        """Add a command from a decorated method"""
        # Use function name as command name
        help_text = method.__doc__.strip() if method.__doc__ else ""
        parser = self.subparsers.add_parser(name, help=help_text)

        # Add arguments
        for arg_def in method._cli_args:
            if arg_def.flag:
                # Boolean flag
                parser.add_argument(
                    f"--{arg_def.name}",
                    action="store_true",
                    default=arg_def.default,
                    help=arg_def.help,
                )
            else:
                # Optional argument (including ... sentinel)
                default_value = None if arg_def.default is ... else arg_def.default
                parser.add_argument(
                    f"--{arg_def.name}",
                    type=arg_def.type,
                    default=default_value,
                    help=arg_def.help,
                )

    def run(self):
        """Parse arguments and run command"""
        args = self.parser.parse_args()

        if not args.command:
            self.parser.print_help()
            return

        # Create app instance
        app = self.app_class()

        # Find and call method
        method = getattr(app, args.command, None)
        if not method:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)

        # Build kwargs from parsed args
        kwargs = {}
        for arg_def in method._cli_args:
            value = getattr(args, arg_def.name, arg_def.default)
            if value is ...:
                value = None
            kwargs[arg_def.name] = value

        # Call method
        method(**kwargs)


# --- Main Application ---


class Jig:
    """jig - Simple deployment tool for Together AI"""

    def __init__(self):
        self.config = Config.load()
        self.state = State.load()

        # Get API key
        self.api_key = os.getenv("TOGETHER_API_KEY", "")
        if not self.api_key:
            print("ERROR: TOGETHER_API_KEY must be set", file=sys.stderr)
            sys.exit(1)

        # Initialize API client
        self.client = APIClient(self.api_key)

        # Get username if needed
        if not self.state.username:
            self.state.username = self.client.get_username()
            self.state.save()

        # Set model name
        if not self.config.model_name:
            self.config.model_name = Path.cwd().name

    def get_image(self, tag: str = "latest") -> str:
        """Get full image name"""
        return f"{REGISTRY_URL}/{self.state.username}/{self.config.model_name}:{tag}"

    def get_image_with_digest(self, tag: str = "latest") -> str:
        """Get full image name tagged with digest"""
        image_name = self.get_image(tag)
        if tag != "latest":
            return image_name

        try:
            # Use docker inspect to get the registry digest from RepoDigests
            result = subprocess.run(
                ["docker", "inspect", "--format={{index .RepoDigests 0}}", image_name],
                capture_output=True,
                text=True,
                check=True,
            )

            image_url = result.stdout.strip()
            if not image_url or image_url == "<no value>":
                raise RuntimeError(
                    f"No registry digest found for {image_name}. "
                    "Make sure the image was pushed to registry first."
                )

            return image_url


        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to get digest for {image_name}: {e.stderr.strip() if e.stderr else 'Docker command failed'}"
            )

    @command()
    def init(self):
        """Initialize jig configuration"""
        pyproject = Path("pyproject.toml")

        if pyproject.exists():
            print("pyproject.toml already exists")
            return

        # Create minimal pyproject.toml
        content = """[project]
name = "my-model"
version = "0.1.0"
dependencies = [
    "torch",
    "transformers",
]

[tool.jig]
model_name = "my-model"

[tool.jig.image]
python_version = "3.11"
system_packages = ["git", "libglib2.0-0"]
cmd = "python app.py"

[tool.jig.deploy]
name = "my-model"
description = "My model deployment"
gpu_type = "h100-80gb"
gpu_count = 1
"""

        with open(pyproject, "w") as f:
            f.write(content)

        print("\N{CHECK MARK} Created pyproject.toml")
        print("  Edit the configuration and run 'jig deploy'")

    @command()
    def dockerfile(self):
        """Generate Dockerfile"""
        if not GENERATE_DOCKERFILE:
            print(
                "Dockerfile generation disabled (set GENERATE_DOCKERFILE=1 to enable)"
            )
            return

        content = generate_dockerfile(self.config)

        # Write file
        with open(self.config.dockerfile, "w") as f:
            f.write(content)

        print("\N{CHECK MARK} Generated Dockerfile")

    @command(arg("tag", default="latest", help="Image tag"))
    def build(self, tag: str = "latest"):
        """Build container image"""
        image = self.get_image(tag)

        # Check if pyproject.toml is newer than Dockerfile
        if GENERATE_DOCKERFILE:
            pyproject_path = Path("pyproject.toml")
            dockerfile_path = Path(self.config.dockerfile)

            if (
                pyproject_path.exists()
                and dockerfile_path.exists()
                and pyproject_path.stat().st_mtime > dockerfile_path.stat().st_mtime
            ):
                print(
                    "\N{INFORMATION SOURCE} pyproject.toml has changed, regenerating Dockerfile"
                )
                self.dockerfile()

            # Generate Dockerfile if needed
            if not dockerfile_path.exists():
                self.dockerfile()

        build_dir_worker_path = Path("./.sprocket.py")
        try:
            shutil.copy(Path(__file__).parent / "sprocket" / "sprocket.py", build_dir_worker_path)
        except FileNotFoundError:
            pass

        print(f"Building {image}")
        cmd = ["docker", "build", "--platform", "linux/amd64", "-t", image, "."]
        if self.config.dockerfile != "Dockerfile":
            cmd.extend(["-f", self.config.dockerfile])
        if subprocess.run(cmd).returncode != 0:
            raise RuntimeError("Build failed")

        build_dir_worker_path.unlink(missing_ok=True)

        print("\N{CHECK MARK} Built")

    @command(arg("tag", default="latest", help="Image tag"))
    def push(self, tag: str = "latest"):
        """Push image to registry"""
        image = self.get_image(tag)

        # Login
        login_cmd = f"echo {self.api_key} | docker login {REGISTRY_URL} --username user --password-stdin"
        if subprocess.run(login_cmd, shell=True, capture_output=True).returncode != 0:
            raise RuntimeError("Registry login failed")

        print(f"Pushing {image}")
        if subprocess.run(["docker", "push", image]).returncode != 0:
            raise RuntimeError("Push failed")

        print("\N{CHECK MARK} Pushed")

    @command()
    def secrets(self):
        """Manage deployment secrets"""
        # maybe this would be cleaner with a sub-CLI
        parser = argparse.ArgumentParser(prog="jig secrets")
        subparsers = parser.add_subparsers(dest="action", help="Secret actions")

        # set subcommand
        set_parser = subparsers.add_parser("set", help="Set a secret")
        set_parser.add_argument("name", help="Secret name")
        set_parser.add_argument("value", help="Secret value")
        set_parser.add_argument("--description", default="", help="Secret description")

        # unset subcommand
        unset_parser = subparsers.add_parser("unset", help="Remove a secret")
        unset_parser.add_argument("name", help="Secret name to remove")

        # list subcommand
        subparsers.add_parser("list", help="List all secrets")

        # Parse remaining args from sys.argv
        import sys

        # Skip past 'jig secrets' in argv
        args_to_parse = sys.argv[2:] if len(sys.argv) > 2 else []
        args = parser.parse_args(args_to_parse)

        if not args.action:
            parser.print_help()
            return

        if args.action == "set":
            self._set_secret(args.name, args.value, args.description)
        elif args.action == "unset":
            self._unset_secret(args.name)
        elif args.action == "list":
            self._list_secrets()

    def _set_secret(self, name: str, value: str, description: str):
        """Set secret for the deployment"""
        deployment_secret_name = f"{self.config.model_name}-{name}"
        secret_data = {
            "name": deployment_secret_name,
            "description": description,
            "value": value,
        }
        try:
            # patch the secret if it exists already
            self.client.request("GET", f"/v1/secrets/{deployment_secret_name}")
            self.client.request(
                "PATCH", f"/v1/secrets/{deployment_secret_name}", json=secret_data
            )
            print(f"\N{CHECK MARK} Updated secret: '{name}'")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                print("\N{ROCKET} Creating new secret")
                self.client.request("POST", "/v1/secrets", json=secret_data)
                print(f"\N{CHECK MARK} Created secret: {name}")
            else:
                raise
        self.state.secrets[name] = deployment_secret_name
        self.state.save()

    def _unset_secret(self, name: str):
        """Unset the secret for the deployment"""
        # FIXME: also delete secret from remote
        if self.state.secrets.pop(name, ""):
            self.state.save()
            print("\N{CHECK MARK} Removed secret from deployment")
        else:
            print(f"Secret {name} is not set")

    def _list_secrets(self):
        """List all secrets for deployment"""
        print(
            f"\N{INFORMATION SOURCE} Following secrets are mapped to deployment {self.config.model_name}"
        )
        for secret_name in self.state.secrets.keys():
            print(f"  - Secret '{secret_name}'")

    @command(
        arg("tag", default="latest", help="Image tag"),
        arg("build_only", type=bool, default=False, help="Build and push only"),
        arg("image", default=..., help="Use existing image (skip build/push)"),
    )
    def deploy(
        self, tag: str = "latest", build_only: bool = False, image: Optional[str] = None
    ):
        """Deploy model"""
        if image:
            # Use provided image, skip build/push
            deployment_image = image
        else:
            # Build and push
            self.build(tag)
            self.push(tag)

            # Get image url pinning to digest
            deployment_image = self.get_image_with_digest(tag)

        if build_only:
            print("\N{CHECK MARK} Build complete (--build-only)")
            return

        deploy_data = {
            "name": self.config.model_name,
            "description": self.config.deploy.description,
            "image": deployment_image,
            "min_replicas": self.config.deploy.min_replicas,
            "max_replicas": self.config.deploy.max_replicas,
            "port": self.config.deploy.port,
            "gpu_type": self.config.deploy.gpu_type,
            "gpu_count": self.config.deploy.gpu_count,
            "cpu": self.config.deploy.cpu,
            "memory": self.config.deploy.memory,
            "autoscaling": self.config.deploy.autoscaling,
        }
        # allow not setting health_check_path
        if self.config.health_check_path:
            deploy_data["health_check_path"] = self.config.health_check_path
        if self.config.deploy.command:
            deploy_data["command"] = self.config.deploy.command

        # Add environment variables
        env_vars = [
            {"name": k, "value": v}
            for k, v in self.config.deploy.environment_variables.items()
        ]
        env_vars.append({"name": "TOGETHER_API_BASE_URL", "value": API_URL})
        if "TOGETHER_API_KEY" not in self.state.secrets:
            self._set_secret("TOGETHER_API_KEY", self.api_key, "Auth key for queue API")

        for name, secret_id in self.state.secrets.items():
            env_vars.append({"name": name, "value_from_secret": secret_id})

        deploy_data["environment_variables"] = env_vars

        # Always use model name for deployment operations
        print(deploy_data)
        print(f"Deploying model: {self.config.model_name}")

        # Try to update first, fallback to create if not found
        try:
            self.client.request(
                "PATCH",
                f"/v1/deployments/{self.config.model_name}",
                json=deploy_data,
            )
            print("\N{CHECK MARK} Updated deployment")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                # Create new deployment
                print("\N{ROCKET} Creating new deployment")
                data = self.client.request("POST", "/v1/deployments", json=deploy_data)
                print(f"\N{CHECK MARK} Deployed: {self.config.model_name}")
                return data
            else:
                raise

    @command()
    def status(self):
        """Get deployment status"""
        data = self.client.request("GET", f"/v1/deployments/{self.config.model_name}")
        pprint(data, indent_guides=False)

    @command()
    def logs(self):
        """Get deployment logs"""
        data = self.client.request(
            "GET", f"/v1/deployments/{self.config.model_name}/logs"
        )
        if data and "lines" in data:
            for line in data["lines"]:
                print(line)
        else:
            print("No logs available")

    @command()
    def destroy(self):
        """Destroy deployment"""
        self.client.request("DELETE", f"/v1/deployments/{self.config.model_name}")
        print(f"\N{WASTEBASKET} Destroyed {self.config.model_name}")

    @command(
        arg("prompt", default=None, help="Job prompt"),
        arg("payload", default=None, help="Job payload JSON"),
        arg(
            "watch", type=bool, default=False, help="Watch job status until completion"
        ),
    )
    def submit(
        self,
        prompt: Optional[str] = None,
        payload: Optional[str] = None,
        watch: bool = False,
    ):
        """Submit a job to the deployment"""
        if not prompt and not payload:
            print("ERROR: Either --prompt or --payload required", file=sys.stderr)
            sys.exit(1)

        job_payload = json.loads(payload) if payload else {"prompt": prompt}

        data = self.client.request(
            "POST",
            "/v1/videos/generations",
            json={
                "model": f"{self.config.model_name}",
                "payload": job_payload,
                "priority": 1,
            },
        )

        print("\N{CHECK MARK} Submitted job")
        pprint(data, indent_guides=False)

        if watch and data and "requestId" in data:
            print(f"\nWatching job {data['requestId']}...")
            self._watch_job_status(data["requestId"])

    def _watch_job_status(self, request_id: str):
        """Watch job status until completion"""
        import time

        last_status = None
        while True:
            try:
                data = self.client.request(
                    "GET",
                    f"/v1/videos/status?request_id={request_id}&model={self.config.model_name}",
                )

                current_status = data.get("status", "")
                if current_status != last_status:
                    pprint(data, indent_guides=False)
                    last_status = current_status

                if current_status in ["done", "failed", "finished", "error"]:
                    break

                time.sleep(1)

            except KeyboardInterrupt:
                print(f"\nStopped watching {request_id}")
                break

    @command(arg("request_id", help="Job request ID"))
    def job_status(self, request_id: str):
        """Get status of a specific video job"""
        data = self.client.request(
            "GET",
            f"/v1/videos/status?request_id={request_id}&model={self.config.model_name}",
        )
        pprint(data, indent_guides=False)

    @command()
    def queue_status(self):
        """Get queue status for the deployment"""
        data = self.client.request(
            "GET", f"/internal/v1/queue/status?model={self.config.model_name}"
        )
        pprint(data, indent_guides=False)


def main():
    """Main entry point"""
    try:
        cli = CLI(Jig)
        cli.run()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except requests.HTTPError as e:
        print(f"API Error: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response.content:
            print(e.response.text, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
