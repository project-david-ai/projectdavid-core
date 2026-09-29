import inspect

from src.api.entities_api.orchestration.mixins.delegation_mixin import DelegationMixin
from src.api.entities_api.orchestration.workers.base_workers.qwen_base import (
    QwenBaseWorker,
)


def test_qwen_provider_exception_propagates():
    source = inspect.getsource(QwenBaseWorker.stream)

    pos = source.rfind("except Exception as exc:")
    assert pos >= 0

    tail = source[pos:]

    assert 'LOG.error(f"Stream Exception: {exc}")' in tail
    assert "\n            raise" in tail or "\n        raise" in tail

    assert 'yield json.dumps({"type": "error", "content": str(exc)' not in tail


def test_failed_worker_marks_delegation_error():
    source = inspect.getsource(DelegationMixin.handle_delegate_research_task)

    assert 'if getattr(event, "status", None) == "failed":' in source

    assert "execution_had_error = True" in source
    assert "worker_failure_reason" in source


def test_worker_run_is_not_force_completed():
    source = inspect.getsource(DelegationMixin.handle_delegate_research_task)

    assert "ephemeral_run.id, StatusEnum.completed.value" not in source


def test_partial_worker_text_cannot_be_success_after_failure():
    source = inspect.getsource(DelegationMixin.handle_delegate_research_task)

    assert "[DELEGATE_FAILURE]" in source

    assert "discarding %d chars of partial stream content" in source

    failure_pos = source.index("if execution_had_error:")
    success_pos = source.index("[DELEGATE_SUCCESS]")

    assert failure_pos < success_pos


def test_terminal_delegation_status_is_honest():
    source = inspect.getsource(DelegationMixin.handle_delegate_research_task)

    assert '"Delegation failed."' in source

    assert '"completed" if not execution_had_error else "error"' in source
