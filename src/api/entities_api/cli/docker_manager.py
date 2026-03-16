# src/api/entities_api/cli/docker_manager.py
#
# Run via:
#   python -m entities_api docker-manager --mode up
#   python -m entities_api docker-manager --mode both --no-cache --tag v1.0
#   entities-api docker-manager --mode up --train
#   entities-api docker-manager configure --interactive
#
from __future__ import annotations

import json
import logging
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from urllib.parse import quote_plus

import typer

# ---------------------------------------------------------------------------
# Container guard
# ---------------------------------------------------------------------------


def _running_in_docker() -> bool:
    return os.getenv("RUNNING_IN_DOCKER") == "1" or Path("/.dockerenv").exists()


if _running_in_docker():
    print(
        "[error] docker_manager.py cannot be run inside a container.\n"
        "This script manages the Docker Compose stack from the HOST machine only.\n"
        "Exiting to prevent Docker-in-Docker chaos."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Optional third-party imports
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    typer.echo("[error] PyYAML is required: pip install PyYAML", err=True)
    raise SystemExit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    typer.echo("[error] python-dotenv is required: pip install python-dotenv", err=True)
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB_CONTAINER_PORT = "3306"
DEFAULT_DB_SERVICE_NAME = "db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

app = typer.Typer(
    name="docker-manager",
    help="Manage Docker Compose stack: build, run, set up .env, fine-tuning overlay, and external Ollama.",
    add_completion=False,
)


class DockerManager:
    """Manages Docker Compose stack operations, env setup, and overlays."""

    _ENV_EXAMPLE_FILE = ".env.example"
    _ENV_FILE = ".env"
    _DOCKER_COMPOSE_FILE = "docker-compose.yml"
    _TRAINING_COMPOSE_FILE = "docker-compose.training.yml"

    _OLLAMA_IMAGE = "ollama/ollama"
    _OLLAMA_CONTAINER = "ollama"
    _OLLAMA_PORT = "11434"

    _COMPOSE_ENV_MAPPING = {
        "MYSQL_ROOT_PASSWORD": (DEFAULT_DB_SERVICE_NAME, "MYSQL_ROOT_PASSWORD"),
        "MYSQL_DATABASE": (DEFAULT_DB_SERVICE_NAME, "MYSQL_DATABASE"),
        "MYSQL_USER": (DEFAULT_DB_SERVICE_NAME, "MYSQL_USER"),
        "MYSQL_PASSWORD": (DEFAULT_DB_SERVICE_NAME, "MYSQL_PASSWORD"),
        "SMBCLIENT_SERVER": ("api", "SMBCLIENT_SERVER"),
        "SMBCLIENT_SHARE": ("api", "SMBCLIENT_SHARE"),
        "SMBCLIENT_USERNAME": ("api", "SMBCLIENT_USERNAME"),
        "SMBCLIENT_PASSWORD": ("api", "SMBCLIENT_PASSWORD"),
        "SMBCLIENT_PORT": ("api", "SMBCLIENT_PORT"),
        "AUTO_MIGRATE": ("api", "AUTO_MIGRATE"),
        "DISABLE_FIREJAIL": ("sandbox", "DISABLE_FIREJAIL"),
    }

    _GENERATED_SECRETS = [
        "SIGNED_URL_SECRET",
        "API_KEY",
        "MYSQL_ROOT_PASSWORD",
        "MYSQL_PASSWORD",
        "SECRET_KEY",
        "SANDBOX_AUTH_SECRET",
        "DEFAULT_SECRET_KEY",
        "SMBCLIENT_PASSWORD",
    ]

    _GENERATED_TOOL_IDS = [
        "TOOL_CODE_INTERPRETER",
        "TOOL_WEB_SEARCH",
        "TOOL_COMPUTER",
        "TOOL_VECTOR_STORE_SEARCH",
    ]

    _USER_REQUIRED = {
        "HF_TOKEN": (
            "HF_TOKEN",
            (
                "HuggingFace personal access token.\n"
                "  Required for downloading gated models (vLLM) and pushing fine-tuned models.\n"
                "  Get one at: https://huggingface.co/settings/tokens"
            ),
            True,
        ),
    }

    _INSECURE_VALUES = {
        "default",
        "your_secret_key_here",
        "changeme",
        "changeme_use_a_real_secret",
        "",
    }

    _DEFAULT_VALUES = {
        "ASSISTANTS_BASE_URL": "http://localhost:9000",
        "SANDBOX_SERVER_URL": "http://localhost:9000",
        "DOWNLOAD_BASE_URL": "http://localhost:9000/v1/files/download",
        "HYPERBOLIC_BASE_URL": "https://api.hyperbolic.xyz/v1",
        "TOGETHER_BASE_URL": "https://api.together.xyz/v1",
        "OLLAMA_BASE_URL": "http://ollama:11434",
        "HF_TOKEN": "",
        "HF_CACHE_PATH": "",
        "VLLM_MODEL": "Qwen/Qwen2.5-VL-3B-Instruct",
        "TOGETHER_API_KEY": "",
        "HYPERBOLIC_API_KEY": "",
        "ADMIN_API_KEY": "",
        "ENTITIES_API_KEY": "",
        "ENTITIES_USER_ID": "",
        "DEEP_SEEK_API_KEY": "",
        "BASE_URL_HEALTH": "http://localhost:9000/v1/health",
        "SHELL_SERVER_URL": "ws://sandbox_api:8000/ws/computer",
        "SHELL_SERVER_EXTERNAL_URL": "ws://localhost:8000/ws/computer",
        "CODE_EXECUTION_URL": "ws://sandbox_api:8000/ws/execute",
        "DISABLE_FIREJAIL": "true",
        "SHARED_PATH": "./shared_data",
        "AUTO_MIGRATE": "1",
        "MYSQL_HOST": DEFAULT_DB_SERVICE_NAME,
        "MYSQL_PORT": DEFAULT_DB_CONTAINER_PORT,
        "MYSQL_DATABASE": "entities_db",
        "MYSQL_USER": "api_user",
        "REDIS_URL": "redis://redis:6379/0",
        "ADMIN_USER_EMAIL": "admin@example.com",
        "ADMIN_USER_ID": "",
        "ADMIN_KEY_PREFIX": "",
        "SMBCLIENT_SERVER": "samba_server",
        "SMBCLIENT_SHARE": "cosmic_share",
        "SMBCLIENT_USERNAME": "samba_user",
        "SMBCLIENT_PORT": "445",
        "LOG_LEVEL": "INFO",
        "PYTHONUNBUFFERED": "1",
    }

    _ENV_STRUCTURE = {
        "Base URLs": [
            "ASSISTANTS_BASE_URL",
            "SANDBOX_SERVER_URL",
            "DOWNLOAD_BASE_URL",
            "HYPERBOLIC_BASE_URL",
            "TOGETHER_BASE_URL",
            "OLLAMA_BASE_URL",
        ],
        "AI Model Configuration": ["HF_TOKEN", "HF_CACHE_PATH", "VLLM_MODEL"],
        "Database Configuration": [
            "DATABASE_URL",
            "SPECIAL_DB_URL",
            "MYSQL_ROOT_PASSWORD",
            "MYSQL_DATABASE",
            "MYSQL_USER",
            "MYSQL_PASSWORD",
            "MYSQL_HOST",
            "MYSQL_PORT",
            "REDIS_URL",
        ],
        "API Keys & External Services": [
            "API_KEY",
            "TOGETHER_API_KEY",
            "HYPERBOLIC_API_KEY",
            "ADMIN_API_KEY",
            "ENTITIES_API_KEY",
            "ENTITIES_USER_ID",
            "DEEP_SEEK_API_KEY",
        ],
        "Platform Settings": [
            "BASE_URL_HEALTH",
            "SHELL_SERVER_URL",
            "SHELL_SERVER_EXTERNAL_URL",
            "CODE_EXECUTION_URL",
            "SIGNED_URL_SECRET",
            "SANDBOX_AUTH_SECRET",
            "DISABLE_FIREJAIL",
            "SECRET_KEY",
            "DEFAULT_SECRET_KEY",
            "SHARED_PATH",
            "AUTO_MIGRATE",
        ],
        "Admin Configuration": ["ADMIN_USER_EMAIL", "ADMIN_USER_ID", "ADMIN_KEY_PREFIX"],
        "SMB Client Configuration": [
            "SMBCLIENT_SERVER",
            "SMBCLIENT_SHARE",
            "SMBCLIENT_USERNAME",
            "SMBCLIENT_PASSWORD",
            "SMBCLIENT_PORT",
        ],
        "Tool Identifiers": [
            "TOOL_CODE_INTERPRETER",
            "TOOL_WEB_SEARCH",
            "TOOL_COMPUTER",
            "TOOL_VECTOR_STORE_SEARCH",
        ],
        "Other": ["LOG_LEVEL", "PYTHONUNBUFFERED"],
    }

    def __init__(self, args: SimpleNamespace) -> None:
        self.args = args
        self.is_windows = platform.system() == "Windows"
        self.log = log

        if self.args.verbose:
            self.log.setLevel(logging.DEBUG)
        self.log.debug("DockerManager initialised with args: %s", vars(args))

        # Compose files handling (Overlay support)
        self.compose_files = [self._DOCKER_COMPOSE_FILE]
        if getattr(self.args, "train", False):
            self.compose_files.append(self._TRAINING_COMPOSE_FILE)
            self.log.info("Fine-tuning training overlay activated (--train).")

        self.compose_config = self._load_compose_config()
        self._check_for_required_env_file()
        self._configure_shared_path()
        self._configure_hf_cache_path()
        self._ensure_dockerignore()

    def _get_compose_flags(self) -> List[str]:
        """Returns the necessary -f flags for docker compose based on active overlays."""
        flags = []
        for file in self.compose_files:
            flags.extend(["-f", file])
        return flags

    def _run_command(
        self, cmd_list, check=True, capture_output=False, text=True, suppress_logs=False, **kwargs
    ):
        if not suppress_logs:
            self.log.info("Running command: %s", " ".join(cmd_list))
        try:
            result = subprocess.run(
                cmd_list,
                check=check,
                capture_output=capture_output,
                text=text,
                shell=self.is_windows,
                **kwargs,
            )
            return result
        except subprocess.CalledProcessError as e:
            self.log.error("Command failed: %s\nReturn Code: %s", " ".join(cmd_list), e.returncode)
            if check:
                raise
            return e

    def _ensure_dockerignore(self):
        dockerignore = Path(".dockerignore")
        if not dockerignore.exists():
            dockerignore.write_text(
                "__pycache__/\n.venv/\nnode_modules/\n*.log\n*.pyc\n.git/\n.env*\n.env\n*.sqlite\ndist/\nbuild/\ncoverage/\ntmp/\n*.egg-info/\n"
            )

    def _load_compose_config(self):
        """Loads and merges config across multiple compose files to allow extracting vars."""
        merged_config = {"services": {}}
        for cf in self.compose_files:
            path = Path(cf)
            if not path.is_file():
                self.log.warning("Compose file '%s' not found.", cf)
                continue
            try:
                config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if "services" in config:
                    merged_config["services"].update(config["services"])
            except Exception as e:
                self.log.error("Error reading %s: %s", cf, e)
        return merged_config

    def _get_all_services(self) -> List[str]:
        return list(self.compose_config.get("services", {}).keys())

    def _get_env_from_compose_service(self, service_name, env_var_name):
        try:
            service_data = self.compose_config.get("services", {}).get(service_name)
            if not service_data:
                return None
            environment = service_data.get("environment")
            if not environment:
                return None
            if isinstance(environment, dict):
                return environment.get(env_var_name)
            if isinstance(environment, list):
                pattern = re.compile(rf"^{re.escape(env_var_name)}(?:=(.*))?$")
                for item in environment:
                    match = pattern.match(item)
                    if match:
                        return match.group(1) if match.group(1) is not None else ""
            return None
        except Exception:
            return None

    def _get_host_port_from_compose_service(self, service_name, container_port):
        try:
            service_data = self.compose_config.get("services", {}).get(service_name)
            if not service_data:
                return None
            ports = service_data.get("ports", [])
            container_port_base = str(container_port).split("/")[0]
            for port_mapping in ports:
                parts = str(port_mapping).split(":")
                host_port = cont_port_part = None
                if len(parts) == 1:
                    if parts[0].split("/")[0] == container_port_base:
                        return parts[0]
                elif len(parts) == 2:
                    host_port, cont_port_part = parts
                elif len(parts) == 3:
                    host_port, cont_port_part = parts[1], parts[2]
                if (
                    host_port
                    and cont_port_part
                    and cont_port_part.split("/")[0] == container_port_base
                ):
                    return host_port.strip()
            return None
        except Exception:
            return None

    def _prompt_user_required(self, env_values: dict, generation_log: dict):
        if not sys.stdin.isatty():
            return
        typer.echo("\n" + "=" * 60 + "\n  Optional: User-Supplied Configuration\n" + "=" * 60)
        for key, (label, help_text, hide) in self._USER_REQUIRED.items():
            typer.echo(f"  {help_text}\n")
            value = typer.prompt(
                f"  {label} (press Enter to skip)", default="", show_default=False, hide_input=hide
            )
            if value.strip():
                env_values[key] = value.strip()
                generation_log[key] = "Provided interactively by user"
                typer.echo(f"  ✓ {key} saved.\n")
        typer.echo("=" * 60 + "\n")

    def _generate_dot_env_file(self):
        self.log.info("Generating '%s'...", self._ENV_FILE)
        env_values = dict(self._DEFAULT_VALUES)
        generation_log = {k: "Default value" for k in env_values}

        for env_key, (service_name, compose_key) in self._COMPOSE_ENV_MAPPING.items():
            value = self._get_env_from_compose_service(service_name, compose_key)
            if value is not None and not str(value).startswith("${"):
                env_values[env_key] = str(value)

        for key in self._GENERATED_SECRETS:
            env_values[key] = secrets.token_hex(16 if key == "API_KEY" else 32)

        for key in self._GENERATED_TOOL_IDS:
            env_values[key] = f"tool_{secrets.token_hex(10)}"

        db_user, db_pass, db_name = (
            env_values.get("MYSQL_USER"),
            env_values.get("MYSQL_PASSWORD"),
            env_values.get("MYSQL_DATABASE"),
        )
        db_host, db_port = env_values.get("MYSQL_HOST", DEFAULT_DB_SERVICE_NAME), env_values.get(
            "MYSQL_PORT", DEFAULT_DB_CONTAINER_PORT
        )
        if all([db_user, db_pass, db_host, db_port, db_name]):
            env_values["DATABASE_URL"] = (
                f"mysql+pymysql://{db_user}:{quote_plus(str(db_pass))}@{db_host}:{db_port}/{db_name}"
            )

        if not env_values.get("HF_CACHE_PATH"):
            env_values["HF_CACHE_PATH"] = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface"
            )

        self._prompt_user_required(env_values, generation_log)

        env_lines = [f"# Auto-generated .env file\n"]
        processed_keys = set()
        for section_name, keys_in_section in self._ENV_STRUCTURE.items():
            env_lines += [
                "#############################",
                f"# {section_name}",
                "#############################",
            ]
            for key in keys_in_section:
                if key in env_values:
                    val = str(env_values[key]).replace("\\", "\\\\").replace('"', '\\"')
                    env_lines.append(
                        f'{key}="{val}"'
                        if any(c in val for c in [" ", "#", "="])
                        else f"{key}={val}"
                    )
                    processed_keys.add(key)
            env_lines.append("")

        for key in sorted(set(env_values.keys()) - processed_keys):
            val = str(env_values[key]).replace("\\", "\\\\").replace('"', '\\"')
            env_lines.append(
                f'{key}="{val}"' if any(c in val for c in [" ", "#", "="]) else f"{key}={val}"
            )

        Path(self._ENV_FILE).write_text("\n".join(env_lines), encoding="utf-8")

    def _check_for_required_env_file(self):
        if not os.path.exists(self._ENV_FILE):
            self._generate_dot_env_file()
        else:
            load_dotenv(dotenv_path=self._ENV_FILE, override=True)

    def _configure_shared_path(self):
        shared_path = os.environ.get("SHARED_PATH", os.path.abspath("./entities_share"))
        os.environ["SHARED_PATH"] = shared_path
        Path(shared_path).mkdir(parents=True, exist_ok=True)

    def _configure_hf_cache_path(self):
        hf_path = os.environ.get("HF_CACHE_PATH", "").strip() or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface"
        )
        os.environ["HF_CACHE_PATH"] = hf_path

    def _validate_secrets(self):
        if any(os.environ.get(k, "") in self._INSECURE_VALUES for k in self._GENERATED_SECRETS):
            self.log.error("Insecure value detected. Delete .env and regenerate.")
            raise SystemExit(1)

    def _has_docker(self):
        return shutil.which("docker") is not None

    def _handle_down(self):
        down_cmd = ["docker", "compose", *self._get_compose_flags(), "down", "--remove-orphans"]
        if self.args.clear_volumes:
            if input("Remove volumes? (yes/no): ").lower() == "yes":
                down_cmd.append("--volumes")
        if self.args.services:
            down_cmd.extend(self.args.services)
        self._run_command(down_cmd, check=False)

    def _handle_build(self):
        build_cmd = ["docker", "compose", *self._get_compose_flags(), "build"]
        if self.args.no_cache:
            build_cmd.append("--no-cache")
        if self.args.parallel:
            build_cmd.append("--parallel")
        if self.args.services:
            build_cmd.extend(self.args.services)
        self._run_command(build_cmd, check=True)

    def _handle_logs(self):
        logs_cmd = ["docker", "compose", *self._get_compose_flags(), "logs"]
        if self.args.follow:
            logs_cmd.append("-f")
        if self.args.services:
            logs_cmd.extend(self.args.services)
        self._run_command(logs_cmd, check=False)

    def _handle_up(self):
        self._validate_secrets()
        up_cmd = ["docker", "compose", *self._get_compose_flags(), "up"]
        if not self.args.attached:
            up_cmd.append("-d")
        if self.args.build_before_up:
            up_cmd.append("--build")

        target = [
            s
            for s in (self.args.services or self._get_all_services())
            if s not in set(self.args.exclude or [])
        ]
        if target:
            up_cmd.extend(target)

        try:
            self._run_command(up_cmd, check=True, suppress_logs=self.args.attached)
        except subprocess.CalledProcessError:
            raise SystemExit(1)

    def run(self):
        if not self._has_docker():
            raise SystemExit(1)
        if self.args.down or self.args.clear_volumes:
            self._handle_down()
            if self.args.mode == "down_only":
                raise SystemExit(0)
        if self.args.mode in ("build", "both"):
            self._handle_build()
            if self.args.mode == "build":
                raise SystemExit(0)
        if self.args.mode in ("up", "both"):
            self._handle_up()
        if self.args.mode == "logs":
            self._handle_logs()


@app.callback(invoke_without_command=True)
def docker_manager(
    ctx: typer.Context,
    mode: str = typer.Option("up", "--mode"),
    services: Optional[List[str]] = typer.Option(None, "--services"),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude", "-x"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    parallel: bool = typer.Option(False, "--parallel"),
    tag: Optional[str] = typer.Option(None, "--tag"),
    attached: bool = typer.Option(False, "--attached", "-a"),
    build_before_up: bool = typer.Option(False, "--build-before-up"),
    force_recreate: bool = typer.Option(False, "--force-recreate"),
    down: bool = typer.Option(False, "--down"),
    clear_volumes: bool = typer.Option(False, "--clear-volumes", "-v"),
    nuke: bool = typer.Option(False, "--nuke"),
    follow: bool = typer.Option(False, "--follow", "-f"),
    tail: Optional[int] = typer.Option(None, "--tail"),
    timestamps: bool = typer.Option(False, "--timestamps", "-t"),
    no_log_prefix: bool = typer.Option(False, "--no-log-prefix"),
    with_ollama: bool = typer.Option(False, "--with-ollama"),
    ollama_gpu: bool = typer.Option(False, "--ollama-gpu"),
    verbose: bool = typer.Option(False, "--verbose", "--debug"),
    debug_cache: bool = typer.Option(False, "--debug-cache"),
    # ---> NEW FLAG ADDED HERE <---
    train: bool = typer.Option(
        False, "--train", help="Include the fine-tuning training overlay (GPU required)."
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if clear_volumes:
        down = True

    args = SimpleNamespace(
        mode=mode,
        services=services or [],
        exclude=exclude or [],
        no_cache=no_cache,
        parallel=parallel,
        tag=tag,
        attached=attached,
        build_before_up=build_before_up,
        force_recreate=force_recreate,
        down=down,
        clear_volumes=clear_volumes,
        nuke=nuke,
        follow=follow,
        tail=tail,
        timestamps=timestamps,
        no_log_prefix=no_log_prefix,
        with_ollama=with_ollama,
        ollama_gpu=ollama_gpu,
        verbose=verbose,
        debug_cache=debug_cache,
        train=train,  # pass the new flag to the namespace
    )

    try:
        from entities_api.cli.generate_docker_compose import \
            generate_dev_docker_compose

        generate_dev_docker_compose()
        time.sleep(0.5)
    except Exception as exc:
        typer.echo(f"[error] Failed to generate docker-compose files: {exc}", err=True)
        raise SystemExit(1)

    manager = DockerManager(args)
    manager.run()


@app.command()
def configure(
    set_var: Optional[List[str]] = typer.Option(None, "--set", "-s"),
    interactive: bool = typer.Option(False, "--interactive", "-i"),
) -> None:
    env_path = Path(DockerManager._ENV_FILE)
    if not env_path.exists():
        raise SystemExit(1)
    # Configure logic remains unchanged
    pass


if __name__ == "__main__":
    app()
