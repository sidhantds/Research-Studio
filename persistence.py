"""Save and restore a browser session's research cases, so a page refresh doesn't lose them.

Each browser session gets a random id that lives in the URL (?sid=...). After every run the
cases are written as JSON to SESSIONS_DIR/<sid>.json; a refresh keeps the URL, so the app reads
the file back. JSON, not pickle: loading a file can never run code.

Anyone with the URL can open that session, like an unlisted link. Files on Streamlit Community
Cloud survive refreshes but not app restarts.
"""

import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from agent import Answer, ToolStep

SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", ".sessions"))
PERSISTED_KEYS = ["dossiers", "runs", "pending_saves", "saved_files", "usage", "active"]
_SID = re.compile(r"^[0-9a-f]{32}$")


def valid_sid(sid) -> bool:
    """Only our own random hex ids: nothing in the URL can point outside SESSIONS_DIR."""
    return isinstance(sid, str) and bool(_SID.match(sid))


def encode_dossier(d: dict) -> dict:
    out = dict(d)
    out["evidence"] = [asdict(s) for s in d.get("evidence", [])]
    out["turns"] = [{**t, "card": t["card"].model_dump() if t.get("card") is not None else None,
                     "fresh": False} for t in d.get("turns", [])]
    return out


def decode_dossier(d: dict) -> dict:
    out = dict(d)
    out["evidence"] = [ToolStep(**s) for s in d.get("evidence", [])]
    out["turns"] = [{**t, "card": Answer.model_validate(t["card"]) if t.get("card") is not None else None}
                    for t in d.get("turns", [])]
    return out


def encode_state(state: dict) -> dict:
    data = {k: state[k] for k in PERSISTED_KEYS if k in state}
    data["dossiers"] = [encode_dossier(d) for d in data.get("dossiers", [])]
    return data


def decode_state(data: dict) -> dict:
    state = {k: data[k] for k in PERSISTED_KEYS if k in data}
    state["dossiers"] = [decode_dossier(d) for d in data.get("dossiers", [])]
    if state.get("active") is not None and not 0 <= state["active"] < len(state["dossiers"]):
        state["active"] = None
    return state


def save(sid: str, state: dict) -> None:
    if not valid_sid(sid):
        return
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(encode_state(state), ensure_ascii=False, default=str)
    # Write to a temp file, then swap it in: a crash mid-write never leaves a half-written session.
    fd, tmp = tempfile.mkstemp(dir=SESSIONS_DIR, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp, SESSIONS_DIR / f"{sid}.json")


def load(sid: str) -> dict | None:
    path = SESSIONS_DIR / f"{sid}.json"
    if not valid_sid(sid) or not path.is_file():
        return None
    try:
        return decode_state(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, TypeError, KeyError):
        return None  # an unreadable file from an older version: start fresh rather than crash
