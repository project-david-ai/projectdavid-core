# src/api/entities_api/utilities/level3_utils.py
import json

from projectdavid import StatusEvent


def create_status_payload(
    run_id: str, tool_name: str, message: str, state: str = "running"
) -> StatusEvent:

    return StatusEvent(
        run_id=run_id,
        status=state,
        tool=tool_name,
        message=message,
    )


def NEW_DOES_NOT_WORK_create_status_payload(
    run_id: str, tool_name: str, message: str, state: str = "running"
) -> str:

    event = StatusEvent(
        run_id=run_id,
        status=state,
        tool=tool_name,
        message=message,
    )

    # Dumps the object's internal dictionary
    return json.dumps(event.__dict__)
