"""Session endpoints: start (identity form) and clear."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException

from api.container import AppContainer
from api.dependencies import get_container
from api.schemas import SessionClearResponse, SessionStartRequest, SessionStartResponse
from core.interfaces import ContactInfo, Turn
from memory.identity import make_user_id

logger = logging.getLogger(__name__)
router = APIRouter(tags=["session"])


def _turns_to_dicts(turns: list[Turn]) -> list[dict]:
    return [
        {"session_id": t.session_id, "user_query": t.user_query,
         "assistant_response": t.assistant_response, "created_at": str(t.created_at)}
        for t in turns
    ]


@router.post("/session/start", response_model=SessionStartResponse)
def start_session(request: SessionStartRequest,
                  container: AppContainer = Depends(get_container)):
    """Handles four cases:
    A) Page refresh with a session already bound to a known user -> resume + return their chats.
    B) Identity entered in the welcome form -> find or create profile, bind a fresh session.
    C) Returning guest -> resume their anonymous session (UI loads its own history).
    D) Brand-new guest.
    """
    db = container.database

    # A) page refresh - session already bound to a known user
    if request.session_id:
        bound_user = db.get_session_user(request.session_id)
        if bound_user:
            profile = db.get_profile(bound_user)
            return SessionStartResponse(
                session_id=request.session_id,
                recognized=True,
                name=profile.name if profile else None,
                email=profile.email if profile else None,
                past_turns=_turns_to_dicts(db.get_user_turns(bound_user, limit=12)),
            )

    # B) identity entered in the welcome form -> find or create profile, bind a fresh session
    if request.email or request.phone:
        user_id = (db.find_user_by_contact(email=request.email, phone=request.phone)
                   or make_user_id(request.email, request.phone))
        existed = db.get_profile(user_id) is not None
        db.upsert_profile(user_id, ContactInfo(name=request.name, email=request.email,
                                               phone=request.phone))
        session_id = f"s-{uuid.uuid4().hex[:8]}"
        db.bind_session(session_id, user_id)
        profile = db.get_profile(user_id)
        return SessionStartResponse(
            session_id=session_id,
            recognized=existed,
            name=profile.name if profile else None,
            email=profile.email if profile else None,
            past_turns=_turns_to_dicts(db.get_user_turns(user_id, limit=12)),
        )

    # C) returning guest - resume their anonymous session
    if request.session_id and db.get_history(request.session_id, limit=1):
        return SessionStartResponse(session_id=request.session_id, recognized=False)

    # D) brand-new guest
    return SessionStartResponse(session_id=f"s-{uuid.uuid4().hex[:8]}", recognized=False)


@router.delete("/session/{session_id}", response_model=SessionClearResponse)
def clear_session(session_id: str, container: AppContainer = Depends(get_container)):
    """Clear short-term memory; long-term memory is saved BEFORE clearing."""
    memory = container.session_manager.get_memory(session_id)
    container.identity.flush_summary(session_id, memory)
    cleared = container.session_manager.clear_session(session_id)
    return SessionClearResponse(session_id=session_id, cleared=cleared)
