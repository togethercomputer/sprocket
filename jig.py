#!/usr/bin/env python3
"""jig - Simple deployment tool for Together AI"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "rich"]
# ///
# pyright: reportPrivateImportUsage=false, reportMissingImports=false

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

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
    from rich.pretty import pprint  # type: ignore
except ImportError:
    try:
        from pip._vendor.rich.pretty import pprint  # type: ignore
    except ImportError:

        def pprint(data: Any, **kwargs: Any) -> None:
            print(json.dumps(data, indent=2))

# --- Configuration ---

TOGETHER_ENV = os.getenv("TOGETHER_ENV", "prod")
if TOGETHER_ENV == "prod":
    API_URL = "api.together.ai"
    REGISTRY_URL = "registry.together.xyz"
elif TOGETHER_ENV == "qa":
    API_URL = "api.qa.together.ai"
    REGISTRY_URL = "registry.t6r-ai.dev"
elif TOGETHER_ENV == "dev":
    API_URL = os.getenv("TOGETHER_API_URL", "")
    REGISTRY_URL = os.getenv("TOGETHER_REGISTRY_URL", "")
    assert API_URL and REGISTRY_URL, "API_URL and REGISTRY_URL must be set in dev mode"
else:
    print("ERROR: unknown together env", TOGETHER_ENV)
    sys.exit(1)

GENERATE_DOCKERFILE = os.getenv("GENERATE_DOCKERFILE", "0") != "0"
DEBUG = os.getenv("TOGETHER_DEBUG", "").strip()[:1] in ("y", "1", "t")


@dataclass
class ImageConfig:
    """Container image configuration from pyproject.toml"""

    python_version: str = "3.11"  # need docstring gen here
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
    """Main configuration from jig.toml or pyproject.toml"""

    model_name: Optional[str] = None
    dockerfile: str = "Dockerfile"
    image: ImageConfig = field(default_factory=ImageConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    _path: Path = Path("pyproject.toml")  # config path or cwd # tweak

    @classmethod
    def find(cls, config_path: Optional[str] = None, init: bool = False) -> "Config":
        "Find specified config_path, pyproject.toml, or jig.toml"
        if config_path:
            if not (found_path := Path(config_path)).exists():
                print(
                    f"ERROR: Configuration file not found: {config_path}",
                    file=sys.stderr,
                )
                sys.exit(1)
            return cls.load(tomllib.load(found_path.open("rb")), found_path)

        if (jigfile := Path("jig.toml")).exists():
            return cls.load(tomllib.load(jigfile.open("rb")), jigfile)

        if (pyproject_path := Path("pyproject.toml")).exists():
            data = tomllib.load(pyproject_path.open("rb"))
            if "tool" in data and "jig" in data["tool"]:
                return cls.load(data, pyproject_path)

        if init:
            return cls()
        print(
            "ERROR: No pyproject.toml or jig.toml found, use --config to specify a config path.",
            file=sys.stderr,
        )
        sys.exit(1)

    @classmethod
    def load(cls, data: dict, path: Path) -> "Config":
        """Load configuration. Useful for manually creating configs"""
        is_pyproject = path.name == "pyproject.toml"

        jig_config = data.get("tool", {}).get("jig", {}) if is_pyproject else data

        name = jig_config.get("name")
        if name is None:
            if is_pyproject:
                name = data.get("project", {}).get("name", "")
            else:
                name = path.resolve().parent.name
                print(
                    f"\N{PACKAGE} Name not set in config file or pyproject.toml - defaulting to {name}"
                )

        if autoscaling := jig_config.get("autoscaling", {}):
            # TODO: validate autoscaling once there are profiles other than QueueBacklogPerWorker
            # maybe this should be in tool.jig.deploy.autoscaling directly
            autoscaling["model"] = name
            jig_config["deploy"]["autoscaling"] = autoscaling

        return cls(
            image=ImageConfig.from_dict(jig_config.get("image", {})),
            deploy=DeployConfig.from_dict(jig_config["deploy"]),
            dockerfile=jig_config.get("dockerfile", "Dockerfile"),
            model_name=name,
            _path=path,
        )


# --- State Management ---


@dataclass
class State:
    """Persistent state"""

    _config_dir: Path
    username: Optional[str] = None
    secrets: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, config_dir: Path) -> "State":
        path = config_dir / ".jig.json"
        try:
            return cls(_config_dir=config_dir, **json.load(open(path)))
        except FileNotFoundError:
            return cls(_config_dir=config_dir)

    def save(self) -> None:
        path = self._config_dir / ".jig.json"
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        json.dump(data, open(path, "w"), indent=2)


# --- API Client and git/docker helper  ---


class APIClient:
    """Together AI API client"""

    def __init__(self, api_key: str) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def request(self, method: str, endpoint: str, **kwargs: Any) -> Optional[dict]:
        """Make API request with error handling"""
        url = f"https://{API_URL}{endpoint}"
        if DEBUG:
            print(method, url)
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None

    def get_username(self) -> str:
        """Get username from proof-data endpoint"""
        response = self.request("GET", "/api/user/proof-data")
        assert response
        # Currently returns project ID as lowercase
        return response["projectId"].lower()


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    "run process with defaults"
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


# --- Container Operations ---


def generate_dockerfile(config: Config) -> str:
    """Generate Dockerfile from config"""
    # Packages installed in both builder and runner
    apt = ""
    if config.image.system_packages:
        sys_pkgs = " ".join(config.image.system_packages or [])
        apt = f"""RUN --mount=type=cache,target=/var/cache/apt \\")
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive \
  apt-get install -y --no-install-recommends {sys_pkgs} && \
  apt-get clean && rm -rf /var/lib/apt/lists/*
"""
    # Environment section
    env = "\n".join(f"ENV {k}={v}" for k, v in config.image.environment.items())
    if env:
        env += "\n"

    # Run commands
    run = "\n".join(f"RUN {cmd}" for cmd in config.image.run)
    if run:
        run += "\n"

    # Files
    copy = "\n".join(f"COPY {file} {file}" for file in get_files_to_copy(config))

    return f"""
# Build stage
FROM python:{config.image.python_version} AS builder

{apt}
# Grab UV to install python packages
COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --compile-bytecode .

# Final stage - slim image
FROM python:{config.image.python_version}-slim

{apt}
COPY --from=builder /usr/local/lib/python{config.image.python_version} /usr/local/lib/python{config.image.python_version}
COPY --from=builder /usr/local/bin /usr/local/bin

# Tini for proper signal handling
COPY --from=krallin/ubuntu-tini:latest /usr/local/bin/tini /tini
ENTRYPOINT ["/tini", "--"]

{env}
{run}
WORKDIR /app
{copy}
# this is temporarily needed if building from a monorepo
RUN --mount=type=bind,source=.,target=/src cp /src/.worker.p* worker.py 2>/dev/null || true
# this tag will set the X-Worker-Version header, used for rollout monitoring
RUN --mount=type=bind,source=.,target=/src git -C /src describe --tags --exact-match > VERSION

CMD {json.dumps(shlex.split(config.image.cmd))}"""


def get_files_to_copy(config: Config) -> list[str]:
    """Get list of files to copy"""
    files = set(config.image.copy)
    if config.image.auto_include_git:
        try:
            # Check if repo is clean
            if run(["git", "status", "--porcelain"]).stdout.strip():
                raise RuntimeError(
                    "Git repository has uncommitted changes: auto_include_git not allowed."
                )
            git_files = run(["git", "ls-files"]).stdout.strip().split("\n")
            files.update(f for f in git_files if f and f != ".")
        except subprocess.CalledProcessError:
            pass  # Not a git repo or git not available

    # Never allow bare "."
    if "." in files:
        raise ValueError("Copying '.' is not allowed. Please enumerate specific files.")

    return sorted(files)


# --- CLI Framework ---


class arg:
    """Argument definition for CLI commands"""

    def __init__(
        self, name: str, type: type = str, default: Any = None, help: str = ""
    ) -> None:
        self.name = name
        self.type = type
        self.default = default
        self.help = help
        # Determine if this is a flag based on type and default
        self.flag = type is bool and default is False


def command(*args: arg) -> Callable:
    """Decorator for CLI commands"""

    def decorator(func: Callable) -> Callable:
        # Store argument definitions
        func._cli_args = list(args)  # type: ignore
        return func

    return decorator


class CLI:
    """Command line interface handler"""

    def __init__(self, app_class: type):
        self.app_class = app_class
        self.parser = argparse.ArgumentParser(
            description=app_class.__doc__ or "CLI Application"
        )
        # Add global --config argument
        self.parser.add_argument(
            "--config", type=str, help="Configuration file path (overrides default)"
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

    def _add_command(self, name: str, method: Callable) -> None:
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

    def run(self) -> None:
        """Parse arguments and run command"""
        args, _ = self.parser.parse_known_args()

        if not args.command:
            self.parser.print_help()
            return

        # Create app instance
        app = self.app_class(config_path=args.config, init=args.command == "init")

        # Find and call method
        if not (method := getattr(app, args.command, None)):
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)

        # Build kwargs from parsed args
        kwargs = {}
        assert hasattr(method, "_cli_args")
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

    def __init__(self, config_path: Optional[str] = None, init: bool = False) -> None:
        self.config = Config.find(config_path, init=init)
        self.state = State.load(self.config._path)

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
            cmd = ["docker", "inspect", "--format={{index .RepoDigests 0}}", image_name]
            if (image_url := run(cmd).stdout.strip()) and image_url != "<no value>":
                return image_url
        except subprocess.CalledProcessError as e:
            msg = e.stderr.strip() if e.stderr else "Docker command failed"
            raise RuntimeError(f"Failed to get digest for {image_name}: {msg}")
        raise RuntimeError(
            f"No registry digest found for {image_name}. Make sure the image was pushed to registry first."
        )

    @command()
    def init(self) -> None:
        """Initialize jig configuration"""
        if (pyproject := Path("pyproject.toml")).exists():
            print("pyproject.toml already exists")
            return
        # Create minimal pyproject.toml
        content = """[project]
name = "my-model"
version = "0.1.0"
dependencies = ["torch", "transformers"]

[tool.jig.image]
python_version = "3.11"
system_packages = ["git", "libglib2.0-0"]
cmd = "python app.py"

[tool.jig.deploy]
description = "My model deployment"
gpu_type = "h100-80gb"
gpu_count = 1
"""
        open(pyproject, "w").write(content)
        print("\N{CHECK MARK} Created pyproject.toml")
        print("  Edit the configuration and run 'jig deploy'")

    @command()
    def dockerfile(self) -> None:
        """Generate Dockerfile"""
        if not GENERATE_DOCKERFILE:
            print("Set GENERATE_DOCKERFILE=1 to enable dockerfile generation")
        else:
            open(self.config.dockerfile, "w").write(generate_dockerfile(self.config))
            print("\N{CHECK MARK} Generated Dockerfile")

    @command(arg("tag", default="latest", help="Image tag"))
    def build(self, tag: str = "latest") -> None:
        """Build container image"""
        image = self.get_image(tag)

        # Check if pyproject.toml is newer than Dockerfile
        if GENERATE_DOCKERFILE:
            dockerfile_path = Path(self.config.dockerfile)
            if (
                self.config._path
                and self.config._path.exists()
                and dockerfile_path.exists()
                and self.config._path.stat().st_mtime > dockerfile_path.stat().st_mtime
            ):
                msg = f"\N{INFORMATION SOURCE} {self.config._path} has changed, regenerating Dockerfile"
                print(msg)
                self.dockerfile()

            # Generate Dockerfile if needed
            if not dockerfile_path.exists():
                self.dockerfile()

        build_dir_worker_path = Path("./.sprocket.py")
        dst = Path(__file__).parent / "sprocket" / "sprocket.py"
        try:
            shutil.copy(dst, build_dir_worker_path)
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
    def push(self, tag: str = "latest") -> None:
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
    def secrets(self) -> None:
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

        # Parse remaining args from sys.argv, skipping past 'jig secrets' in argv
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

    def _set_secret(self, name: str, value: str, description: str) -> None:
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
            if e.response.status_code != 404:
                raise
            print("\N{ROCKET} Creating new secret")
            self.client.request("POST", "/v1/secrets", json=secret_data)
            print(f"\N{CHECK MARK} Created secret: {name}")

        self.state.secrets[name] = deployment_secret_name
        self.state.save()

    def _unset_secret(self, name: str) -> None:
        """Unset the secret for the deployment"""
        # FIXME: also delete secret from remote
        if self.state.secrets.pop(name, ""):
            self.state.save()
            print("\N{CHECK MARK} Removed secret from deployment")
        else:
            print(f"Secret {name} is not set")

    def _list_secrets(self) -> None:
        """List all secrets for deployment"""
        msg = f"\N{INFORMATION SOURCE} Following secrets are mapped to deployment {self.config.model_name}"
        print(msg)
        for secret_name in self.state.secrets:
            print(f"  - Secret '{secret_name}'")

    @command(
        arg("tag", default="latest", help="Image tag"),
        arg("build_only", type=bool, default=False, help="Build and push only"),
        arg("image", default=..., help="Use existing image (skip build/push)"),
    )
    def deploy(
        self, tag: str = "latest", build_only: bool = False, image: Optional[str] = None
    ) -> Optional[dict]:
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
            return print("\N{CHECK MARK} Build complete (--build-only)")

        deploy_data: "dict[str, Any]" = {
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
        if self.config.deploy.health_check_path:
            deploy_data["health_check_path"] = self.config.deploy.health_check_path
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
            response = self.client.request(
                "PATCH",
                f"/v1/deployments/{self.config.model_name}",
                json=deploy_data,
            )
            print("\N{CHECK MARK} Updated deployment")
        except requests.HTTPError as e:
            if e.response.status_code != 404:
                raise
            # Create new deployment
            print("\N{ROCKET} Creating new deployment")
            response = self.client.request("POST", "/v1/deployments", json=deploy_data)
            print(f"\N{CHECK MARK} Deployed: {self.config.model_name}")
        return response

    @command()
    def status(self) -> None:
        """Get deployment status"""
        response = self.client.request(
            "GET", f"/v1/deployments/{self.config.model_name}"
        )
        pprint(response, indent_guides=False)

    @command(arg("follow", type=bool, default=False, help="Follow log output"))
    def logs(self, follow: bool = False) -> None:
        """Get deployment logs"""
        if not follow:
            response = self.client.request(
                "GET", f"/v1/deployments/{self.config.model_name}/logs"
            )
            if response and "lines" in response:
                for log_line in response["lines"]:
                    print(log_line, flush=True)
            else:
                print("No logs available")
            return
        url = f"https://{API_URL}/v1/deployments/{self.config.model_name}/logs?follow=true"
        try:
            response = self.client.session.get(url, stream=True, timeout=None)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    for log_line in json.loads(line).get("lines", []):
                        print(log_line, flush=True)
        except KeyboardInterrupt:
            print("\nStopped following logs")
        except Exception as e:
            print(f"\nConnection ended: {e}")

    @command()
    def destroy(self) -> None:
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
    ) -> None:
        """Submit a job to the deployment"""
        if not prompt and not payload:
            print("ERROR: Either --prompt or --payload required", file=sys.stderr)
            sys.exit(1)

        request = {
            "model": f"{self.config.model_name}",
            "payload": json.loads(payload) if payload else {"prompt": prompt},
            "priority": 1,
        }
        response = self.client.request("POST", "/v1/videos/generations", json=request)
        print("\N{CHECK MARK} Submitted job")
        pprint(response, indent_guides=False)

        if watch and response and "requestId" in response:
            print(f"\nWatching job {response['requestId']}...")
            self._watch_job_status(response["requestId"])

    def _watch_job_status(self, request_id: str) -> None:
        """Watch job status until completion"""
        last_status = None
        while True:
            try:
                response = self.client.request(
                    "GET",
                    f"/v1/videos/status?request_id={request_id}&model={self.config.model_name}",
                )
                current_status = (response or {}).get("status", "")
                if current_status != last_status:
                    pprint(response, indent_guides=False)
                    last_status = current_status

                if current_status in ["done", "failed", "finished", "error"]:
                    break

                time.sleep(1)

            except KeyboardInterrupt:
                print(f"\nStopped watching {request_id}")
                break

    @command(arg("request_id", help="Job request ID"))
    def job_status(self, request_id: str) -> None:
        """Get status of a specific video job"""
        response = self.client.request(
            "GET",
            f"/v1/videos/status?request_id={request_id}&model={self.config.model_name}",
        )
        pprint(response, indent_guides=False)

    @command()
    def queue_status(self) -> None:
        """Get queue status for the deployment"""
        response = self.client.request(
            "GET", f"/internal/v1/queue/status?model={self.config.model_name}"
        )
        pprint(response, indent_guides=False)


def main() -> None:
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
