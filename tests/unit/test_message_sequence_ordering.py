import inspect

from projectdavid_orm import Message

from src.api.entities_api.services.message_service import MessageService


def test_message_model_exposes_canonical_sequence():
    column = Message.__table__.c.sequence_no

    assert column.name == "sequence_no"
    assert column.nullable is False
    assert column.unique is True
    assert column.server_default is not None


def test_message_service_uses_sequence_for_chronology():
    source = inspect.getsource(MessageService)

    assert "Message.created_at.asc()" not in source
    assert "Message.created_at.desc()" not in source

    assert "Message.sequence_no.asc()" in source
