from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session

from src.api.entities_api.models.models import AccountErasure, Base
from src.api.entities_api.services.erasure_state import (
    COMPLETE,
    PENDING,
    erasure_state,
    erasures,
    subject_hash,
)


def test_erasure_table_uses_authoritative_metadata():
    assert erasures is AccountErasure.__table__
    assert erasures.metadata is Base.metadata


def test_subject_hash_is_stable():
    first = subject_hash("user_test")
    second = subject_hash("user_test")

    assert first == second
    assert len(first) == 64
    assert first != "user_test"
    assert first != subject_hash("user_other")


def test_erasure_state_lifecycle():
    engine = create_engine("sqlite:///:memory:")

    try:
        erasures.create(engine)

        with Session(engine) as db:
            assert erasure_state(db, "user_test") is None

            db.execute(
                erasures.insert().values(
                    subject_hash=subject_hash("user_test"),
                    state=PENDING,
                    created_at=100,
                )
            )
            db.commit()

            assert erasure_state(db, "user_test") == PENDING
            assert erasure_state(db, "user_other") is None

            db.execute(
                erasures.update()
                .where(erasures.c.subject_hash == subject_hash("user_test"))
                .values(
                    state=COMPLETE,
                    completed_at=200,
                )
            )
            db.commit()

        with Session(engine) as db:
            assert erasure_state(db, "user_test") == COMPLETE

    finally:
        engine.dispose()
