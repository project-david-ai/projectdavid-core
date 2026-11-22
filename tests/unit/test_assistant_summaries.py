"""Assistant summary filtering, ownership, and protected-resource isolation."""

import unittest

from fastapi import HTTPException
from sqlalchemy.engine.create import create_engine
from sqlalchemy.orm import Session

from src.api.entities_api.models.models import Assistant, User
from src.api.entities_api.services.assistants_service import (
    AssistantService,
    SummaryFilters,
)


class AssistantSummaryTests(unittest.TestCase):
    def setUp(self):
        # Core requires an ORM model carrying the protected-Assistant flag.
        # Do not synthesize this column in tests: doing so would hide a
        # dependency mismatch between the installed ORM and Core.
        if "is_protected" not in Assistant.__table__.c:
            self.fail(
                "Installed Assistant ORM model lacks is_protected. "
                "Align the local projectdavid-orm dependency with the "
                "version required by Core before running these tests."
            )

        self.engine = create_engine("sqlite://")

        User.__table__.create(self.engine)
        Assistant.__table__.create(self.engine)

        self.associations = Assistant.users.property.secondary
        self.associations.create(self.engine)

        self.db = Session(self.engine)

        self.db.execute(
            User.__table__.insert(),
            [{"id": "u1"}, {"id": "u2"}],
        )

        for assistant_id, owner_id, deleted_at, is_protected in [
            ("a", "u1", None, False),
            ("b", "u1", None, False),
            ("legacy", None, None, False),
            ("foreign", "u2", None, False),
            ("deleted", "u1", 200, False),
            ("protected", "u1", None, True),
        ]:
            self.db.execute(
                Assistant.__table__.insert().values(
                    id=assistant_id,
                    owner_id=owner_id,
                    object="assistant",
                    created_at=100,
                    name=assistant_id,
                    model="test",
                    deleted_at=deleted_at,
                    is_protected=is_protected,
                )
            )

        # These associations exercise:
        # - an assistant already included through canonical ownership;
        # - a legacy assistant without a canonical owner;
        # - an assistant canonically owned by another user;
        # - a soft-deleted assistant;
        # - an owned and associated protected assistant (must stay hidden).
        self.db.execute(
            self.associations.insert(),
            [
                {"user_id": "u1", "assistant_id": assistant_id}
                for assistant_id in (
                    "a",
                    "legacy",
                    "foreign",
                    "deleted",
                    "protected",
                )
            ],
        )
        self.db.commit()

    def tearDown(self):
        if hasattr(self, "db"):
            self.db.close()
        if hasattr(self, "engine"):
            self.engine.dispose()

    def test_ownership_soft_deletion_and_no_duplicates(self):
        page = AssistantService.list_summaries(
            self.db,
            "u1",
            SummaryFilters(),
        )

        self.assertEqual(
            [item.id for item in page.items],
            ["legacy", "b", "a"],
        )
        self.assertFalse(page.has_more)
        self.assertIsNone(page.next_cursor)

    def test_pagination_with_equal_timestamps(self):
        first = AssistantService.list_summaries(
            self.db,
            "u1",
            SummaryFilters(limit=2),
        )

        self.assertEqual(
            [item.id for item in first.items],
            ["legacy", "b"],
        )
        self.assertTrue(first.has_more)
        self.assertIsNotNone(first.next_cursor)

        second = AssistantService.list_summaries(
            self.db,
            "u1",
            SummaryFilters(
                limit=2,
                cursor=first.next_cursor,
            ),
        )

        self.assertEqual(
            [item.id for item in second.items],
            ["a"],
        )
        self.assertFalse(second.has_more)
        self.assertIsNone(second.next_cursor)

    def test_empty_history(self):
        page = AssistantService.list_summaries(
            self.db,
            "nobody",
            SummaryFilters(),
        )

        self.assertEqual(
            page.model_dump(),
            {
                "items": [],
                "next_cursor": None,
                "has_more": False,
            },
        )

    def test_invalid_cursor(self):
        with self.assertRaises(HTTPException) as error:
            AssistantService.list_summaries(
                self.db,
                "u1",
                SummaryFilters(cursor="bad!"),
            )

        self.assertEqual(error.exception.status_code, 422)

    def test_assistant_id_filter(self):
        page = AssistantService.list_summaries(
            self.db,
            "u1",
            SummaryFilters(assistant_id="b"),
        )

        self.assertEqual(
            [item.id for item in page.items],
            ["b"],
        )

    def test_protected_assistant_is_hidden_even_from_owner(self):
        page = AssistantService.list_summaries(
            self.db,
            "u1",
            SummaryFilters(),
        )
        self.assertNotIn("protected", [item.id for item in page.items])

        direct = AssistantService.list_summaries(
            self.db,
            "u1",
            SummaryFilters(assistant_id="protected"),
        )
        self.assertEqual(direct.items, [])
        self.assertFalse(direct.has_more)


if __name__ == "__main__":
    unittest.main()
