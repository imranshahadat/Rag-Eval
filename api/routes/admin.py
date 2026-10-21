"""Owner-facing endpoints: contacts, profiles, history."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.container import AppContainer
from api.dependencies import get_container
from api.schemas import ContactResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin"])


@router.get("/contact/{session_id}", response_model=ContactResponse)
def get_contact(session_id: str, container: AppContainer = Depends(get_container)):
    """Contact info for a session (resolved via session -> user profile)."""
    user_id = container.database.get_session_user(session_id)
    profile = container.database.get_profile(user_id) if user_id else None
    if not profile:
        raise HTTPException(status_code=404, detail="No contact info for this session")
    return ContactResponse(session_id=session_id, name=profile.name, email=profile.email,
                           phone=profile.phone, updated_at=profile.updated_at)


@router.get("/profiles")
def get_all_profiles(container: AppContainer = Depends(get_container)):
    """Every known user and their long-term memory."""
    profiles = container.database.get_all_profiles()
    return {"count": len(profiles), "profiles": [p.__dict__ for p in profiles]}


@router.get("/history/{session_id}")
def get_history(session_id: str, limit: int = 20,
                container: AppContainer = Depends(get_container)):
    """Full message log (user message, assistant message, timestamp) for a session."""
    turns = container.database.get_history(session_id, limit=limit)
    return {
        "session_id": session_id,
        "turns": [
            {"user_query": t.user_query, "assistant_response": t.assistant_response,
             "created_at": str(t.created_at)}
            for t in turns
        ],
    }
