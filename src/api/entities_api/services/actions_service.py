# src/api/entities_api/services/actions_service.py
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import Action, Run
from src.api.entities_api.utils.conversion_utils import datetime_to_iso

validator = ValidationInterface()
logging_utility = LoggingUtility()


class ActionService:
    def __init__(self):
        logging_utility.info("ActionService initialized.")

    @staticmethod
    def _normalize_confidence(payload: Optional[Dict[str, Any]]) -> Optional[float]:
        """
        Extracts and sanitizes confidence score from decision payload.
        Handles: 0.95, 95, "0.95", "95%", None.
        Returns a float between 0.0 and 1.0.
        """
        if not payload:
            return None

        raw = payload.get("confidence")
        if raw is None:
            return None

        try:
            # Convert string to float if necessary (stripping % if present)
            if isinstance(raw, str):
                raw = float(raw.replace("%", "").strip())

            val = float(raw)

            # Heuristic: If > 1.0, assume it's a percentage (e.g. 95 -> 0.95)
            if val > 1.0:
                val = val / 100.0

            # Clamp to 0.0 - 1.0 range just to be safe
            return max(0.0, min(1.0, val))

        except (ValueError, TypeError):
            # Log this mismatch if needed, but return None or 0.0 to prevent crashing
            return 0.0

    def update_action_stream_state(self, action_id: str, state: dict):
        with SessionLocal() as db:
            action = db.query(Action).get(action_id)
            if not action:
                raise HTTPException(status_code=404, detail="Action not found")

            if action.meta_data is None:
                action.meta_data = {}

            current_meta = dict(action.meta_data) if action.meta_data else {}
            current_meta["stream_state"] = {
                "buffer": state.get("buffer", []),
                "received_lines": state.get("received_lines", 0),
                "last_update": datetime.utcnow().isoformat(),
            }
            action.meta_data = current_meta
            db.commit()

    def update_action_output(
        self, action_id: str, new_content: str, is_partial: bool = True
    ):
        with SessionLocal() as db:
            action = db.query(Action).get(action_id)
            if not action:
                raise HTTPException(status_code=404, detail="Action not found")

            if not action.result or isinstance(action.result, str):
                action.result = {
                    "full_output": "",
                    "partials": [],
                    "status": "in_progress",
                }

            current_result = dict(action.result)
            if is_partial:
                current_result["partials"].append(
                    {
                        "function_call": new_content,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            else:
                current_result["full_output"] = new_content
                current_result["status"] = "completed"
                action.status = "completed"
                action.processed_at = datetime.utcnow()

            action.result = current_result
            db.commit()

    def create_action(
        self, action_data: validator.ActionCreate
    ) -> validator.ActionRead:
        """
        Creates a new action.
        COMPATIBILITY FIX: Now persists tool_name, decision_payload, and confidence_score.
        """
        logging_utility.info(
            "Creating action for tool: %s, run_id: %s",
            action_data.tool_name,
            action_data.run_id,
        )

        with SessionLocal() as db:
            try:
                # [FIX 1] Move Logic Inside Try Block
                # If this fails (e.g. AttributeError), it will now be caught by the generic Exception handler below
                # instead of returning a raw 500.

                # [FIX 2] Correct Attribute Name
                # Changed 'action_data.decision_payload' to 'action_data.decision' to match the Pydantic model
                # field used in the Action constructor below.
                calculated_confidence = self._normalize_confidence(action_data.decision)

                new_action_id = UtilsInterface.IdentifierService.generate_action_id()
                new_action = Action(
                    id=new_action_id,
                    run_id=action_data.run_id,
                    triggered_at=datetime.now(),
                    expires_at=action_data.expires_at,
                    function_args=action_data.function_args,
                    status=action_data.status or "pending",
                    tool_call_id=action_data.tool_call_id,
                    tool_name=action_data.tool_name,
                    turn_index=action_data.turn_index or 0,
                    # --- [NEW] TELEMETRY FIELDS ---
                    decision_payload=action_data.decision,  # The full JSON (Why)
                    confidence_score=calculated_confidence,  # The Index (How sure)
                    # ------------------------------
                )

                db.add(new_action)
                db.commit()
                db.refresh(new_action)

                return validator.ActionRead(
                    id=new_action.id,
                    run_id=new_action.run_id,
                    tool_call_id=new_action.tool_call_id,
                    tool_name=new_action.tool_name,
                    status=new_action.status,
                    result=new_action.result,
                    triggered_at=datetime_to_iso(new_action.triggered_at),
                    turn_index=new_action.turn_index,
                )
            except IntegrityError as e:
                db.rollback()
                logging_utility.error(
                    "IntegrityError during action creation: %s", str(e)
                )
                raise HTTPException(status_code=400, detail="Invalid action data")
            except Exception as e:
                db.rollback()
                # Use traceback to make future debugging easier
                import traceback

                logging_utility.error(
                    "Unexpected error in CreateAction: %s\n%s",
                    str(e),
                    traceback.format_exc(),
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Server Error during action creation: {str(e)}",
                )

    def get_action(self, action_id: str) -> validator.ActionRead:
        """Retrieve an action by its ID with the new tool_name field."""
        with SessionLocal() as db:
            action = db.query(Action).filter(Action.id == action_id).first()
            if not action:
                raise HTTPException(
                    status_code=404, detail=f"Action {action_id} not found"
                )

            return validator.ActionRead(
                id=action.id,
                run_id=action.run_id,
                tool_name=action.tool_name,  # COMPATIBILITY FIX
                tool_call_id=action.tool_call_id,
                turn_index=action.turn_index,
                triggered_at=datetime_to_iso(action.triggered_at),
                expires_at=datetime_to_iso(action.expires_at),
                is_processed=action.is_processed,
                processed_at=datetime_to_iso(action.processed_at),
                status=action.status,
                function_args=action.function_args,
                result=action.result,
            )

    def update_action_status(
        self, action_id: str, action_update: validator.ActionUpdate
    ) -> validator.ActionRead:
        with SessionLocal() as db:
            action = db.query(Action).filter(Action.id == action_id).first()
            if not action:
                raise HTTPException(status_code=404, detail="Action not found")

            action.status = action_update.status
            action.result = action_update.result

            if str(action_update.status) == "completed":
                action.is_processed = True
                action.processed_at = datetime.now()

            db.commit()
            db.refresh(action)

            return validator.ActionRead(
                id=action.id,
                run_id=action.run_id,
                tool_call_id=action.tool_call_id,
                tool_name=action.tool_name,  # COMPATIBILITY FIX
                status=action.status,
                result=action.result,
                processed_at=datetime_to_iso(action.processed_at),
            )

    def get_actions_by_status(
        self, run_id: str, status: Optional[str] = "pending"
    ) -> List[validator.ActionRead]:
        with SessionLocal() as db:
            actions = (
                db.query(Action)
                .filter(Action.run_id == run_id, Action.status == status)
                .all()
            )
            return [
                validator.ActionRead(
                    id=action.id,
                    run_id=action.run_id,
                    tool_call_id=action.tool_call_id,
                    tool_name=action.tool_name,  # COMPATIBILITY FIX
                    triggered_at=datetime_to_iso(action.triggered_at),
                    status=action.status,
                    function_args=action.function_args,
                    result=action.result,
                )
                for action in actions
            ]

    def get_pending_actions(self, run_id=None) -> List[Dict[str, Any]]:
        """
        Retrieves pending actions using the Action ORM model directly.
        """
        with SessionLocal() as db:
            target_statuses = ["pending", "action_required", "requires_action"]

            # 1. Select Action objects
            # We use joinedload(Action.run) to efficiently fetch the related Run status in one go
            query = (
                db.query(Action)
                .options(joinedload(Action.run))
                .filter(Action.status.in_(target_statuses))
            )

            # 2. Filter by Run ID if provided
            if run_id:
                query = query.filter(Action.run_id == run_id)

            actions = query.all()

            # 3. Convert to Dictionary
            return [
                {
                    "action_id": action.id,
                    "tool_call_id": action.tool_call_id,
                    "action_status": action.status,
                    "function_arguments": action.function_args,  # Automatically handled since column is JSON type
                    "tool_name": action.tool_name,
                    "run_id": action.run_id,
                    # Access the relationship safely
                    "run_status": action.run.status if action.run else None,
                }
                for action in actions
            ]

    def delete_action(self, action_id: str) -> None:
        with SessionLocal() as db:
            action = db.query(Action).filter(Action.id == action_id).first()
            if not action:
                raise HTTPException(status_code=404, detail="Action not found")
            db.delete(action)
            db.commit()
