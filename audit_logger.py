"""
audit_logger.py - Audit Trial Logger

Creates structured audit log entries and writes the complete trial to a permanent JSON file for compliance record-keeping (FCA accountability)

"""

import json
import os
from datetime import datetime


def create_audit_entry(case_id: str, step: str, action: str, details: dict) -> dict:
    """
    Builds one structured audit log entry with a real timestamp.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "case_id": case_id,
        "step": step,
        "action": action,
        "details": details
    }

def write_audit_trail(audit_log: list, case_id: str, output_dir: str = "audit_trails") -> str:
    """
    Writes the complete audit_log (a list of structured entries created by
    create_audit_entry) to a permanent JSON file. Returns the file path so it
    can be referenced in the final report (supports FCA accountability -
    a traceable, timestamped record of every step taken for this case).
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"audit_trail_{case_id}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(audit_log, f, indent=2)

    return filepath