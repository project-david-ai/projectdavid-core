import builtins
import importlib
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from entities_api.__main__ import root
from typer.testing import CliRunner

runner = CliRunner()


def _execution_mocks(*, generator_error=None):
    stack = ExitStack()
    generator = stack.enter_context(
        patch(
            "entities_api.cli.generate_docker_compose.generate_dev_docker_compose",
            side_effect=generator_error,
        )
    )
    manager_type = stack.enter_context(
        patch("entities_api.cli.docker_manager.DockerManager")
    )
    stack.enter_context(patch("entities_api.cli.docker_manager.time.sleep"))
    return stack, generator, manager_type


def test_docker_manager_help_exits_without_execution():
    stack, generator, manager_type = _execution_mocks()
    with stack:
        result = runner.invoke(root, ["docker-manager", "--help"])

    assert result.exit_code == 0
    assert "--mode" in result.output
    generator.assert_not_called()
    manager_type.assert_not_called()


@pytest.mark.parametrize("mode", ["up", "build", "both", "down_only", "logs"])
def test_each_valid_mode_is_accepted(mode):
    stack, generator, manager_type = _execution_mocks()
    with stack:
        result = runner.invoke(root, ["docker-manager", "--mode", mode])

    assert result.exit_code == 0, result.output
    generator.assert_called_once_with()
    manager_type.assert_called_once()
    assert manager_type.call_args.args[0].mode == mode
    manager_type.return_value.run.assert_called_once_with()


def test_invalid_mode_is_rejected_before_execution():
    stack, generator, manager_type = _execution_mocks()
    with stack:
        result = runner.invoke(
            root,
            ["docker-manager", "--mode", "nonsense"],
        )

    assert result.exit_code == 2
    assert "Invalid value for '--mode'" in result.output
    generator.assert_not_called()
    manager_type.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [
        ["docker-manager", "--mode", "--help"],
        ["docker-manager", "--mode", "up", "--help"],
    ],
)
def test_mode_help_never_reaches_execution(arguments):
    stack, generator, manager_type = _execution_mocks()
    with stack:
        result = runner.invoke(root, arguments)

    assert result.exit_code in {0, 2}
    generator.assert_not_called()
    manager_type.assert_not_called()


def test_compose_generator_import_does_not_require_top_level_src(monkeypatch):
    generator_module = importlib.import_module(
        "entities_api.cli.generate_docker_compose"
    )
    real_import = builtins.__import__

    def reject_src_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src" or name.startswith("src."):
            raise AssertionError(f"unexpected top-level src import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    stack, generator, manager_type = _execution_mocks()
    with (
        stack,
        patch.object(
            generator_module,
            "generate_dev_docker_compose",
            generator,
        ),
    ):
        monkeypatch.setattr(builtins, "__import__", reject_src_import)
        result = runner.invoke(root, ["docker-manager", "--mode", "up"])

    assert result.exit_code == 0, result.output
    generator.assert_called_once_with()
    manager_type.return_value.run.assert_called_once_with()


@pytest.mark.parametrize(
    "arguments",
    [
        ["docker-manager", "configure", "--help"],
        ["docker-manager", "bootstrap-admin", "--help"],
    ],
)
def test_existing_subcommand_help_still_parses(arguments):
    stack, generator, manager_type = _execution_mocks()
    with stack:
        result = runner.invoke(root, arguments)

    assert result.exit_code == 0, result.output
    generator.assert_not_called()
    manager_type.assert_not_called()


def test_compose_generation_failure_is_concise_without_verbose():
    stack, _, manager_type = _execution_mocks(
        generator_error=RuntimeError("compose exploded")
    )
    with stack:
        result = runner.invoke(root, ["docker-manager", "--mode", "up"])

    assert result.exit_code == 1
    assert "Failed to generate docker-compose files: compose exploded" in result.output
    assert "Traceback (most recent call last)" not in result.output
    manager_type.assert_not_called()


def test_verbose_compose_generation_failure_includes_traceback():
    stack, _, manager_type = _execution_mocks(
        generator_error=RuntimeError("compose exploded")
    )
    with stack:
        result = runner.invoke(
            root,
            ["docker-manager", "--mode", "up", "--verbose"],
        )

    assert result.exit_code == 1
    assert "Failed to generate docker-compose files: compose exploded" in result.output
    assert "Traceback (most recent call last)" in result.output
    assert "RuntimeError: compose exploded" in result.output
    manager_type.assert_not_called()
