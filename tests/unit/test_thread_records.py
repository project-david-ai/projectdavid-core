import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from projectdavid_common import ValidationInterface

from src.api.entities_api.routers.threads_router import router
from src.api.entities_api.services.threads_service import ThreadService

validator = ValidationInterface()


def _thread(
    thread_id: str,
    *,
    created_at: int,
    meta_data=None,
    owner_id: str = "user_owner",
):
    return SimpleNamespace(
        id=thread_id,
        created_at=created_at,
        meta_data=meta_data if meta_data is not None else {},
        object="thread",
        tool_resources={},
        owner_id=owner_id,
        participants=[],
    )


def _session_for_threads(threads):
    db = MagicMock()
    query = db.query.return_value
    query.join.return_value = query
    query.filter.return_value = query
    query.all.return_value = threads

    context = MagicMock()
    context.__enter__.return_value = db
    context.__exit__.return_value = False
    return db, context


def _session_for_thread_records(records):
    return _session_for_threads(records)


def test_existing_id_list_contract_and_order_are_unchanged():
    threads = [
        _thread("thread_second", created_at=2),
        _thread("thread_first", created_at=1),
    ]
    _, context = _session_for_threads(threads)

    with patch(
        "src.api.entities_api.services.threads_service.SessionLocal",
        return_value=context,
    ):
        result = ThreadService().list_threads_by_user("user_owner")

    assert result == ["thread_second", "thread_first"]


def test_records_endpoint_uses_thread_read_without_detailed_relationships():
    route = next(
        route
        for route in router.routes
        if route.path == "/threads/user/{user_id}/records"
    )

    assert route.response_model == list[validator.ThreadRead]


def test_thread_records_include_metadata_and_retain_query_order():
    threads = [
        _thread(
            "thread_second",
            created_at=2,
            meta_data=json.dumps({"q": {"title": "Second"}}),
        ),
        _thread("thread_first", created_at=1, meta_data={"foo": 1}),
    ]
    _, context = _session_for_thread_records(
        [
            (threads[0], True),
            (threads[1], False),
        ]
    )

    with patch(
        "src.api.entities_api.services.threads_service.SessionLocal",
        return_value=context,
    ):
        result = ThreadService().list_thread_records_by_user("user_owner")

    assert [record.id for record in result] == [
        "thread_second",
        "thread_first",
    ]
    assert result[0].meta_data == {"q": {"title": "Second"}}
    assert result[1].meta_data == {"foo": 1}
    assert result[0].materialized is True
    assert result[1].materialized is False
    assert all(type(record) is validator.ThreadRead for record in result)


def test_both_list_methods_share_the_participant_filtered_query():
    threads = [_thread("thread_authorized", created_at=1)]
    db, context = _session_for_thread_records(
        [
            (threads[0], True),
        ]
    )

    with patch(
        "src.api.entities_api.services.threads_service.SessionLocal",
        return_value=context,
    ):
        records = ThreadService().list_thread_records_by_user("user_owner")

    assert [record.id for record in records] == ["thread_authorized"]
    db.query.return_value.join.assert_called_once()
    db.query.return_value.filter.assert_called_once()


def test_thread_read_materialization_defaults_to_unknown_outside_record_list():
    record = ThreadService()._create_thread_read(
        _thread(
            "thread_unknown",
            created_at=1,
        )
    )

    assert record.materialized is None


def test_metadata_patch_preserves_top_level_and_nested_q_metadata():
    service = ThreadService()
    original_q = {"computer": "something"}
    thread = _thread(
        "thread_1",
        created_at=1,
        meta_data={"foo": 1, "q": original_q},
    )
    db = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = db
    context.__exit__.return_value = False

    with (
        patch(
            "src.api.entities_api.services.threads_service.SessionLocal",
            return_value=context,
        ),
        patch.object(service, "_get_thread_or_404", return_value=thread),
    ):
        result = service.update_thread_metadata(
            "thread_1",
            {
                "q": {
                    "title": "Networking Jobs in Germany",
                    "title_source": "auto",
                }
            },
            user_id="user_owner",
        )

    assert thread.meta_data == {
        "foo": 1,
        "q": {
            "computer": "something",
            "title": "Networking Jobs in Germany",
            "title_source": "auto",
        },
    }
    assert result.meta_data == thread.meta_data


def test_metadata_update_assigns_fresh_json_state():
    service = ThreadService()
    original_metadata = {"q": {"computer": "something"}}
    thread = _thread(
        "thread_1",
        created_at=1,
        meta_data=original_metadata,
    )
    db = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = db
    context.__exit__.return_value = False

    with (
        patch(
            "src.api.entities_api.services.threads_service.SessionLocal",
            return_value=context,
        ),
        patch.object(service, "_get_thread_or_404", return_value=thread),
    ):
        service.update_thread_metadata(
            "thread_1",
            {"q": {"title": "Fresh title"}},
            user_id="user_owner",
        )

    assert thread.meta_data is not original_metadata
    assert thread.meta_data["q"] is not original_metadata["q"]
    assert original_metadata == {"q": {"computer": "something"}}
    db.commit.assert_called_once_with()
