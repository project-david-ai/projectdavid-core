# src/api/entities_api/cli/docker_manager.py
#
# Run via:
#   platform-api docker-manager --mode up
#   platform-api docker-manager --mode both --no-cache --tag v1.0
#   platform-api docker-manager --mode up --services training-api
#   platform-api docker-manager --mode build --services training-api
#   platform-api docker-manager bootstrap-admin
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
    help="Manage Docker Compose stack: build, run, set up .env, and manage services.",
    add_completion=False,
)


class DockerManager:
    """Manages Docker Compose stack operations and env setup."""

    _ENV_FILE = ".env"
    _DOCKER_COMPOSE_FILE = "docker-compose.yml"

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
        "SEARXNG_SECRET_KEY",
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
            "SEARXNG_SECRET_KEY",
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

        self.compose_files = [self._DOCKER_COMPOSE_FILE]

        self.compose_config = self._load_compose_config()
        self._check_for_required_env_file()
        self._configure_shared_path()
        self._configure_hf_cache_path()
        self._ensure_dockerignore()

    def _get_compose_flags(self) -> List[str]:
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
                "__pycache__/\n.venv/\nnode_modules/\n*.log\n*.pyc\n.git/\n"
                ".env*\n.env\n*.sqlite\ndist/\nbuild/\ncoverage/\ntmp/\n*.egg-info/\n"
            )

    def _load_compose_config(self):
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

    def _prompt_user_required(self, env_values: dict, generation_log: dict):
        if not sys.stdin.isatty():
            self.log.warning("Non-interactive environment detected. Prompting skipped.")
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
                generation_log[env_key] = (
                    f"From {self._DOCKER_COMPOSE_FILE} ({service_name}/{compose_key})"
                )

        for key in self._GENERATED_SECRETS:
            env_values[key] = secrets.token_hex(16 if key == "API_KEY" else 32)
            generation_log[key] = "Generated new secret"

        for key in self._GENERATED_TOOL_IDS:
            env_values[key] = f"tool_{secrets.token_hex(10)}"
            generation_log[key] = "Generated new tool ID"

        db_user = env_values.get("MYSQL_USER")
        db_pass = env_values.get("MYSQL_PASSWORD")
        db_name = env_values.get("MYSQL_DATABASE")
        db_host = env_values.get("MYSQL_HOST", DEFAULT_DB_SERVICE_NAME)
        db_port = env_values.get("MYSQL_PORT", DEFAULT_DB_CONTAINER_PORT)

        if all([db_user, db_pass, db_host, db_port, db_name]):
            env_values["DATABASE_URL"] = (
                f"mysql+pymysql://{db_user}:{quote_plus(str(db_pass))}"
                f"@{db_host}:{db_port}/{db_name}"
            )
            generation_log["DATABASE_URL"] = "Constructed from DB components"

        if not env_values.get("HF_CACHE_PATH"):
            env_values["HF_CACHE_PATH"] = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface"
            )

        self._prompt_user_required(env_values, generation_log)

        env_lines = ["# Auto-generated .env file\n"]
        processed_keys: set = set()

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
        self.log.info("Successfully generated '%s'.", self._ENV_FILE)

    def _check_for_required_env_file(self):
        if not os.path.exists(self._ENV_FILE):
            self.log.warning("'%s' missing. Generating...", self._ENV_FILE)
            self._generate_dot_env_file()
        else:
            self.log.info("'%s' exists. Loading.", self._ENV_FILE)
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
        failed = False
        for key in self._GENERATED_SECRETS:
            val = os.environ.get(key, "")
            if val in self._INSECURE_VALUES:
                self.log.error(
                    "Insecure or missing value for '%s'. Delete .env and re-run to regenerate.", key
                )
                failed = True
        if failed:
            raise SystemExit(1)

    def _has_docker(self):
        return shutil.which("docker") is not None

    def _is_container_running(self, container_name):
        if not self._has_docker():
            return False
        try:
            result = self._run_command(
                ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
                capture_output=True,
                check=False,
                suppress_logs=True,
            )
            return result.stdout.strip() == container_name
        except Exception:
            return False

    def _has_nvidia_support(self):
        cmd = shutil.which("nvidia-smi.exe" if self.is_windows else "nvidia-smi") or (
            shutil.which("nvidia-smi") if self.is_windows else None
        )
        if not cmd:
            cmd = shutil.which("nvidia-smi")
        if cmd:
            try:
                self._run_command([cmd], check=True, capture_output=True, suppress_logs=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        return False

    def _start_ollama(self, cpu_only=True):
        if not self._has_docker():
            return False
        container_name = self._OLLAMA_CONTAINER
        if self._is_container_running(container_name):
            self.log.info("External Ollama container '%s' is already running.", container_name)
            return True
        if not self._is_image_present(self._OLLAMA_IMAGE):
            self.log.info("Pulling Ollama image '%s'...", self._OLLAMA_IMAGE)
            try:
                self._run_command(["docker", "pull", self._OLLAMA_IMAGE], check=True)
            except Exception as e:
                self.log.error("Failed to pull Ollama image: %s", e)
                return False
        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-v",
            "ollama:/root/.ollama",
            "-p",
            f"{self._OLLAMA_PORT}:{self._OLLAMA_PORT}",
            "--name",
            container_name,
        ]
        gpu_support_added = False
        if not cpu_only and self._has_nvidia_support():
            cmd.extend(["--gpus", "all"])
            gpu_support_added = True
        cmd.append(self._OLLAMA_IMAGE)
        try:
            self._run_command(cmd, check=True)
            time.sleep(5)
            if self._is_container_running(container_name):
                mode = "GPU" if gpu_support_added else "CPU"
                self.log.info("External Ollama container started in %s mode.", mode)
                return True
            self.log.error("Ollama container failed to start.")
            return False
        except Exception as e:
            self.log.error("Error starting Ollama container: %s", e)
            return False

    def _ensure_ollama(self, opt_in=False, use_gpu=False):
        if not opt_in:
            return True
        self.log.info("--- External Ollama Setup ---")
        if os.path.exists("/.dockerenv") or "DOCKER_HOST" in os.environ:
            self.log.warning("Running inside Docker; skipping external Ollama management.")
            return True
        if platform.system() == "Darwin" and use_gpu:
            self.log.warning(
                "macOS detected; GPU passthrough has limitations. CPU mode recommended."
            )
        success = self._start_ollama(cpu_only=not use_gpu)
        if not success:
            self.log.error("Failed to start the external Ollama container.")
        self.log.info("--- End External Ollama Setup ---")
        return success

    def _run_docker_cache_diagnostics(self):
        self.log.info("--- Docker Cache Diagnostics ---")
        if not self._has_docker():
            return
        try:
            total_size = sum(
                f.stat().st_size
                for f in Path(".").resolve().rglob("*")
                if f.is_file() and not f.is_symlink()
            )
            size_mb = total_size / (1024 * 1024)
            self.log.info("Approximate build context size: %.2f MB", size_mb)
            if size_mb > 500:
                self.log.warning(
                    "Build context > 500 MB. Review .dockerignore to exclude large files."
                )

            config_res = self._run_command(
                ["docker", "compose", *self._get_compose_flags(), "config", "--format", "json"],
                capture_output=True,
                check=True,
                suppress_logs=True,
            )
            service_configs = json.loads(config_res.stdout).get("services", {})
            for svc, cfg in service_configs.items():
                image_name = cfg.get("image")
                if not image_name:
                    continue
                self.log.info("--- History for image '%s' (service: %s) ---", image_name, svc)
                history_res = self._run_command(
                    [
                        "docker",
                        "history",
                        image_name,
                        "--no-trunc=false",
                        "--format",
                        '{{printf "%.12s" .ID}} | {{.Size | printf "%-12s"}} | {{.CreatedBy}}',
                    ],
                    check=False,
                    capture_output=True,
                    suppress_logs=True,
                )
                if history_res.returncode == 0 and history_res.stdout.strip():
                    for line in history_res.stdout.strip().splitlines():
                        self.log.info(line)
        except Exception as e:
            self.log.error("Diagnostics error: %s", e)
        finally:
            self.log.info("--- End Docker Cache Diagnostics ---")

    def _handle_nuke(self):
        self.log.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.log.warning("!!!    NUKE MODE ACTIVATED   !!!")
        self.log.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.log.warning(
            "This will stop all compose services, remove their volumes, AND prune ALL unused Docker data system-wide."
        )
        try:
            confirm = input(">>> Type 'confirm nuke' exactly to proceed: ")
        except EOFError:
            self.log.error("Nuke requires interactive confirmation. Aborting.")
            raise SystemExit(1)
        if confirm.strip() != "confirm nuke":
            self.log.info("Nuke operation cancelled.")
            raise SystemExit(0)
        self.log.info("Proceeding with Docker nuke...")
        self._run_command(
            [
                "docker",
                "compose",
                *self._get_compose_flags(),
                "down",
                "--volumes",
                "--remove-orphans",
            ],
            check=False,
        )
        try:
            self._run_command(
                ["docker", "system", "prune", "-a", "--volumes", "--force"], check=True
            )
        except subprocess.CalledProcessError as e:
            self.log.critical("docker system prune failed (code %s).", e.returncode)
            raise SystemExit(1)
        self.log.info("*** Docker Nuke Complete ***")

    def _handle_down(self):
        down_cmd = [
            "docker",
            "compose",
            *self._get_compose_flags(),
            "down",
            "--remove-orphans",
        ]
        if self.args.clear_volumes:
            try:
                if input("Remove volumes? (yes/no): ").lower().strip() == "yes":
                    down_cmd.append("--volumes")
            except EOFError:
                self.log.error("Volume deletion confirmation requires interactive input. Aborting.")
                raise SystemExit(1)

        if self.args.services:
            down_cmd.extend(self.args.services)
        self._run_command(down_cmd, check=False)

    def _tag_images(self, tag, targeted_services=None):
        if not (tag and self._has_docker()):
            return
        try:
            config_res = self._run_command(
                ["docker", "compose", *self._get_compose_flags(), "config", "--format", "json"],
                capture_output=True,
                check=True,
                suppress_logs=True,
            )
            services_data = json.loads(config_res.stdout).get("services", {})
            for svc_name, svc_config in services_data.items():
                if targeted_services and svc_name not in targeted_services:
                    continue
                image_name = svc_config.get("image")
                if not image_name:
                    continue
                base_image = image_name.split(":")[0]
                new_tag = f"{base_image}:{tag}"
                self.log.info("Tagging: %s  ->  %s", image_name, new_tag)
                self._run_command(
                    ["docker", "tag", image_name, new_tag], check=True, suppress_logs=True
                )
        except Exception as e:
            self.log.error("Error during image tagging: %s", e)

    def _handle_build(self):
        build_cmd = ["docker", "compose", *self._get_compose_flags(), "build"]
        if self.args.no_cache:
            build_cmd.append("--no-cache")
        if self.args.parallel:
            build_cmd.append("--parallel")
        if self.args.services:
            build_cmd.extend(self.args.services)
        try:
            self._run_command(build_cmd, check=True)
            if self.args.tag:
                self._tag_images(self.args.tag, targeted_services=self.args.services or None)
        except subprocess.CalledProcessError as e:
            self.log.critical("Docker build failed (code %s). Check logs above.", e.returncode)
            raise SystemExit(1)

    def _handle_logs(self):
        logs_cmd = ["docker", "compose", *self._get_compose_flags(), "logs"]
        if self.args.follow:
            logs_cmd.append("-f")
        if self.args.tail:
            logs_cmd.extend(["--tail", str(self.args.tail)])
        if self.args.timestamps:
            logs_cmd.append("-t")
        if self.args.no_log_prefix:
            logs_cmd.append("--no-log-prefix")

        if self.args.services:
            logs_cmd.extend(self.args.services)
        self._run_command(logs_cmd, check=False)

    def _handle_up(self):
        self._validate_secrets()

        # -------------------------------------------------------
        # HARD RULE:
        # We ALWAYS explicitly tell compose what services to start
        # so profile services NEVER auto-start.
        # -------------------------------------------------------

        all_services = self.compose_config.get("services", {})
        profile_services = {name for name, cfg in all_services.items() if cfg.get("profiles")}

        # FIX: Bypass the profile hard rule specifically for 'nginx'
        if "nginx" in profile_services:
            profile_services.remove("nginx")

        default_services = [name for name in all_services.keys() if name not in profile_services]

        requested = set(self.args.services or [])
        excluded = set(self.args.exclude or [])

        if requested:
            final_services = requested - excluded
        else:
            final_services = set(default_services) - excluded

        final_services = sorted(final_services)

        if not final_services:
            self.log.error("No services resolved to start.")
            raise SystemExit(1)

        # FIX: Start 'nginx' service explicitly before the others
        if "nginx" in final_services:
            self.log.info("Starting 'nginx' container first...")
            nginx_cmd = ["docker", "compose", *self._get_compose_flags(), "up", "-d"]

            if self.args.build_before_up:
                nginx_cmd.append("--build")

            if self.args.force_recreate:
                nginx_cmd.append("--force-recreate")

            nginx_cmd.append("nginx")

            try:
                self._run_command(nginx_cmd, check=True, suppress_logs=False)
            except subprocess.CalledProcessError:
                raise SystemExit(1)

            # Remove nginx from final batch as it's already up
            final_services.remove("nginx")

        # Start the remaining services
        if final_services:
            up_cmd = ["docker", "compose", *self._get_compose_flags(), "up"]

            if not self.args.attached:
                up_cmd.append("-d")

            if self.args.build_before_up:
                up_cmd.append("--build")

            if self.args.force_recreate:
                up_cmd.append("--force-recreate")

            up_cmd.extend(final_services)

            self.log.info("Starting remaining services: %s", ", ".join(final_services))

            try:
                self._run_command(
                    up_cmd,
                    check=True,
                    suppress_logs=self.args.attached,
                )
            except subprocess.CalledProcessError:
                raise SystemExit(1)

    def _is_service_running(self, service_name: str) -> bool:
        if not self._has_docker():
            return False
        try:
            res = self._run_command(
                [
                    "docker",
                    "compose",
                    *self._get_compose_flags(),
                    "ps",
                    "--services",
                    "--filter",
                    "status=running",
                ],
                capture_output=True,
                check=True,
                suppress_logs=True,
            )
            return service_name in res.stdout.split()
        except Exception:
            return False

    def _ensure_api_running(self, action: str):
        if not self._is_service_running("api"):
            self.log.error(
                "The 'api' service is not running. Start the stack first:\n"
                "  platform-api docker-manager --mode up"
            )
            raise SystemExit(1)

    def exec_bootstrap_admin(self, db_url: Optional[str] = None):
        self._ensure_api_running("bootstrap-admin")

        # Ensure environment variables are loaded
        load_dotenv(dotenv_path=self._ENV_FILE, override=True)
        resolved_db_url = db_url or os.environ.get("DATABASE_URL")

        if not resolved_db_url:
            self.log.error(
                "No DATABASE_URL found. Ensure it is set in .env or pass --db-url explicitly."
            )
            raise SystemExit(1)

        cmd = [
            "docker",
            "compose",
            *self._get_compose_flags(),
            "exec",
            "api",
            "python",
            "/app/src/api/entities_api/cli/bootstrap_admin.py",
            "--db-url",
            resolved_db_url,
        ]

        try:
            self._run_command(cmd, check=True, suppress_logs=True)
            self.log.info(
                "bootstrap-admin finished. "
                "Copy any printed ADMIN_API_KEY — it is required for API-level user provisioning."
            )
        except subprocess.CalledProcessError:
            raise SystemExit(1)

    def run(self):
        if self.args.debug_cache:
            self._run_docker_cache_diagnostics()
            raise SystemExit(0)

        if self.args.nuke:
            self._handle_nuke()
            raise SystemExit(0)

        if not self._has_docker():
            raise SystemExit(1)

        if self.args.with_ollama:
            self._ensure_ollama(opt_in=True, use_gpu=self.args.ollama_gpu)

        if self.args.mode == "down_only":
            self.args.down = True

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
    """Update specific variables in an existing .env without regenerating secrets."""
    env_path = Path(DockerManager._ENV_FILE)
    if not env_path.exists():
        typer.echo(
            f"[error] '{DockerManager._ENV_FILE}' not found. "
            "Run 'docker-manager --mode up' first to generate it.",
            err=True,
        )
        raise SystemExit(1)

    load_dotenv(dotenv_path=env_path, override=True)
    updates: dict = {}

    if set_var:
        for item in set_var:
            if "=" not in item:
                typer.echo(f"[error] Invalid format '{item}'. Expected KEY=VALUE.", err=True)
                raise SystemExit(1)
            key, _, value = item.partition("=")
            updates[key.strip()] = value.strip()

    if interactive:
        typer.echo("\n" + "=" * 60 + "\n  Interactive Configuration\n" + "=" * 60)
        for key, (label, help_text, hide) in DockerManager._USER_REQUIRED.items():
            current = os.environ.get(key, "")
            status = "(currently set)" if current else "(currently blank)"
            typer.echo(f"  {help_text}\n")
            value = typer.prompt(
                f"  {label} {status}",
                default="",
                show_default=False,
                hide_input=hide,
            )
            if value.strip():
                updates[key] = value.strip()
                typer.echo(f"  ✓ {key} will be updated.\n")
        typer.echo("=" * 60 + "\n")

    if not updates:
        typer.echo("Nothing to update. Use --set KEY=VALUE or --interactive.")
        raise SystemExit(0)

    content = env_path.read_text(encoding="utf-8")
    for key, value in updates.items():
        if any(c in value for c in [" ", "#", "="]):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            new_line = f'{key}="{escaped}"'
        else:
            new_line = f"{key}={value}"

        pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
        if pattern.search(content):
            content = pattern.sub(new_line, content)
        else:
            content += f"\n# Added by configure\n{new_line}\n"

    env_path.write_text(content, encoding="utf-8")
    typer.echo(
        f"✓ {len(updates)} variable(s) updated: {', '.join(updates.keys())}\n"
        "Restart the stack for changes to take effect:\n"
        "  platform-api docker-manager --mode up --force-recreate"
    )


@app.command(name="bootstrap-admin")
def bootstrap_admin(
    db_url: Optional[str] = typer.Option(
        None,
        "--db-url",
        help="Override DATABASE_URL for the bootstrap script. Defaults to DATABASE_URL from .env.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    """
    Provision the default admin user inside the running api container.

    The stack must be running before calling this command.
    Copy any printed ADMIN_API_KEY — it is required for API-level user provisioning.

    Safe to re-run: existing users and keys are detected and left untouched.
    """
    args = SimpleNamespace(
        mode="up",
        services=[],
        exclude=[],
        no_cache=False,
        parallel=False,
        tag=None,
        attached=False,
        build_before_up=False,
        force_recreate=False,
        down=False,
        clear_volumes=False,
        nuke=False,
        follow=False,
        tail=None,
        timestamps=False,
        no_log_prefix=False,
        with_ollama=False,
        ollama_gpu=False,
        verbose=verbose,
        debug_cache=False,
    )

    manager = DockerManager(args)
    manager.exec_bootstrap_admin(db_url=db_url)


if __name__ == "__main__":
    app()
