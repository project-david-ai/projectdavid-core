"""Regression tests for selective Docker CI filters."""

from scripts import ci_component_filter as ci


def _toml(
    version="1.61.8",
    dependencies='["fastapi"]',
    extra="",
):
    return (
        '[build-system]\nrequires = ["setuptools"]\n'
        f'\n[project]\nname = "entities_api"\nversion = "{version}"\n'
        f"dependencies = {dependencies}\n{extra}"
    ).encode()


def test_version_only_metadata_change_needs_no_build():
    assert not ci.meaningful_toml_change(
        _toml("1.61.7"),
        _toml("1.61.8"),
    )


def test_comment_only_change_needs_no_build():
    assert not ci.meaningful_toml_change(
        _toml(),
        _toml() + b"\n# comment\n",
    )


def test_dependency_change_triggers_build():
    assert ci.meaningful_toml_change(
        _toml("1.61.7"),
        _toml("1.61.8", '["fastapi", "sqlalchemy"]'),
    )


def test_non_project_version_is_not_ignored():
    old = _toml(extra='\n[tool.example]\nversion = "1"\n')
    new = _toml(extra='\n[tool.example]\nversion = "2"\n')

    assert ci.meaningful_toml_change(old, new)


def test_missing_or_malformed_toml_fails_closed():
    assert ci.meaningful_toml_change(
        None,
        _toml(),
    )

    assert ci.meaningful_toml_change(
        _toml(),
        None,
    )

    assert ci.meaningful_toml_change(
        b"[project\n",
        _toml(),
    )


def test_sync_both_version_numbers_builds_nothing(monkeypatch):
    files = {
        ("base", ci.ROOT_TOML): _toml("1.61.7"),
        ("head", ci.ROOT_TOML): _toml("1.61.8"),
        ("base", ci.SANDBOX_TOML): _toml("1.61.7"),
        ("head", ci.SANDBOX_TOML): _toml("1.61.8"),
    }

    monkeypatch.setattr(
        ci,
        "git_file",
        lambda rev, path: files[rev, path],
    )

    flags = ci.effective_filters(
        {},
        base="base",
        head="head",
        root_toml_changed=True,
        sandbox_toml_changed=True,
    )

    assert flags == {name: False for name in ci.COMPONENTS}


def test_root_dependency_edit_rebuilds_api_only(monkeypatch):
    monkeypatch.setattr(
        ci,
        "git_file",
        lambda rev, path: (
            _toml(dependencies='["fastapi", "pydantic"]') if rev == "head" else _toml()
        ),
    )

    flags = ci.effective_filters(
        {},
        base="base",
        head="head",
        root_toml_changed=True,
        sandbox_toml_changed=False,
    )

    assert {name for name, hit in flags.items() if hit} == {"api"}


def test_sandbox_dependency_edit_rebuilds_sandbox_only(
    monkeypatch,
):
    monkeypatch.setattr(
        ci,
        "git_file",
        lambda rev, path: (
            _toml(dependencies='["fastapi", "websockets"]')
            if rev == "head"
            else _toml()
        ),
    )

    flags = ci.effective_filters(
        {},
        base="base",
        head="head",
        root_toml_changed=False,
        sandbox_toml_changed=True,
    )

    assert {name for name, hit in flags.items() if hit} == {"sandbox"}


def test_changed_source_flags_are_preserved():
    flags = ci.effective_filters(
        {
            "training": True,
            "inference": True,
        },
        base="base",
        head="head",
        root_toml_changed=False,
        sandbox_toml_changed=False,
    )

    assert {name for name, hit in flags.items() if hit} == {"training", "inference"}


def test_unreadable_revision_triggers_safe_build(monkeypatch):
    monkeypatch.setattr(
        ci,
        "git_file",
        lambda rev, path: None,
    )

    flags = ci.effective_filters(
        {},
        base="missing",
        head="head",
        root_toml_changed=True,
        sandbox_toml_changed=False,
    )

    assert flags["api"]
