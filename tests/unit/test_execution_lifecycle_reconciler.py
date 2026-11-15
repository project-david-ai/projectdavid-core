"""Pure safety-contract tests; no database or live services required."""

import ast
import pathlib
import unittest

SOURCE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/api/entities_api/services/execution_lifecycle_reconciler.py"
)
# Extract pure classifier without importing the application or its dependencies.
tree = ast.parse(SOURCE.read_text())
fn = next(
    n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "classify"
)
namespace = {
    "TERMINAL": frozenset({"completed", "failed", "cancelled", "expired"}),
    "ACTIVE_EPOCH": frozenset({"reserved", "running"}),
    "ACTIVE_ACTOR": frozenset({"reserved", "active"}),
}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), namespace)
classify = namespace["classify"]


class SafetyTests(unittest.TestCase):
    def kind(
        self,
        run="completed",
        epoch="completed",
        actor="retired",
        aid="asst_x",
        deleted=True,
        age=90000,
    ):
        return classify(run, epoch, actor, aid, deleted, age)[0]

    def test_clean_with_historical_identity(self):
        self.assertEqual(self.kind(), "CLEAN")

    def test_terminal_active_actor(self):
        self.assertEqual(self.kind(actor="active"), "TERMINAL_LEDGER_REPAIR")

    def test_terminal_running_epoch(self):
        self.assertEqual(self.kind(epoch="running"), "TERMINAL_LEDGER_REPAIR")

    def test_retired_but_assistant_not_deleted(self):
        self.assertEqual(self.kind(deleted=False), "ASSISTANT_DELETION_PENDING")

    def test_missing_assistant(self):
        self.assertEqual(self.kind(deleted=None), "INCONSISTENT")

    def test_conflicting_terminal_outcomes(self):
        self.assertEqual(self.kind(epoch="failed"), "INCONSISTENT")

    def test_user_handoff_never_age_expires(self):
        self.assertEqual(
            self.kind(
                run="action_required", epoch="running", actor="active", age=999999999
            ),
            "AWAITING_USER",
        )

    def test_suspect_does_not_mutate(self):
        self.assertEqual(
            self.kind(run="in_progress", epoch="running", actor="active", age=9999999),
            "SUSPECT_NONTERMINAL",
        )

    def test_nonterminal_healthy(self):
        self.assertEqual(
            self.kind(run="in_progress", epoch="running", actor="active", age=20),
            "HEALTHY",
        )

    def test_nonterminal_retired_inconsistent(self):
        self.assertEqual(self.kind(run="in_progress"), "INCONSISTENT")

    def test_missing_run(self):
        self.assertEqual(self.kind(run="missing"), "INCONSISTENT")

    def test_terminal_reserved_no_identity(self):
        self.assertEqual(
            self.kind(epoch="reserved", actor="reserved", aid=None, deleted=None),
            "TERMINAL_LEDGER_REPAIR",
        )


if __name__ == "__main__":
    unittest.main()
