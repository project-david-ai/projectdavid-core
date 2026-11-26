from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import HTTPException

import src.api.entities_api.services.file_service as file_module
import src.api.entities_api.services.message_service as message_module
from src.api.entities_api.services.message_service import MessageService


def test_attachment_lookup_scopes_file_to_owner_and_non_deleted():
    db = Mock()
    query = db.query.return_value.filter.return_value
    query.first.return_value = None

    assert not MessageService._owned_attachment_exists(db, "file_1", "owner_1")

    file_model = message_module.File
    predicates = db.query.return_value.filter.call_args.args
    assert len(predicates) == 3
    assert predicates[0].compare(file_model.id == "file_1")
    assert predicates[1].compare(file_model.user_id == "owner_1")
    assert predicates[2].compare(file_model.deleted_at.is_(None))


def test_attachment_admission_rejects_unowned_reference(monkeypatch):
    monkeypatch.setattr(
        MessageService, "_owned_attachment_exists", staticmethod(lambda *args: False)
    )

    with pytest.raises(HTTPException) as exc:
        MessageService._assert_attachment_owners(
            Mock(), [{"type": "image", "file_id": "file_1"}], "owner_1"
        )

    assert exc.value.status_code == 404


def test_hydration_never_reads_foreign_or_deleted_attachment(monkeypatch):
    service = MessageService()
    file_service = Mock()
    file_service.get_file_as_base64.return_value = "iVBORw0KGgo="
    monkeypatch.setattr(file_module, "FileService", Mock(return_value=file_service))
    owner_check = Mock(side_effect=[False, False, True])
    monkeypatch.setattr(service, "_owned_attachment_exists", owner_check)

    result = service._hydrate_attachments(
        Mock(),
        "hello",
        [
            {"type": "image", "file_id": "foreign"},
            {"type": "image", "file_id": "deleted"},
            {"type": "image", "file_id": "owned"},
        ],
        user_id="owner_1",
    )

    file_service.get_file_as_base64.assert_called_once_with("owned", user_id="owner_1")
    file_service.get_file_as_base64_internal.assert_not_called()
    assert result == [
        {"type": "text", "text": "hello"},
        {"type": "image", "image": "data:image/png;base64,iVBORw0KGgo="},
    ]


def test_hydration_without_owner_returns_plain_text(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(file_module, "FileService", factory)
    result = MessageService()._hydrate_attachments(
        Mock(), "hello", [{"type": "image", "file_id": "file_1"}], user_id=None
    )

    assert result == "hello"
    factory.assert_not_called()


@pytest.mark.parametrize("internal", [True, False])
def test_formatted_history_propagates_thread_owner(monkeypatch, internal):
    session = MagicMock()
    db = session.__enter__.return_value
    db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
        owner_id="owner_1"
    )
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
        []
    )
    monkeypatch.setattr(message_module, "SessionLocal", lambda: session)

    service = MessageService()
    formatter = Mock(return_value=[])
    monkeypatch.setattr(service, "_format_messages_from_db", formatter)

    if internal:
        assert service.get_formatted_messages_internal("thread_1") == []
    else:
        assert service.get_formatted_messages("thread_1", "owner_1") == []

    formatter.assert_called_once_with(
        [], hydrate_images=True, include_attachments=False, db=db, user_id="owner_1"
    )
