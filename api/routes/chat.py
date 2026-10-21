"""Chat endpoints (non-streaming and streaming)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.container import AppContainer
from api.dependencies import get_container
from api.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, container: AppContainer = Depends(get_container)):
    """Per-session memory, DB persistence, identity resolution, long-term memory."""
    memory = container.session_manager.get_memory(request.session_id)
    try:
        is_first_turn = len(memory.chat_history) == 0

        # 1. Who is this? (extracts contact, binds session, replays cross-session memory)
        profile = container.identity.resolve(request.session_id, request.message, memory)
        user_context = container.identity.build_context(profile, is_first_turn)

        # 2. Answer (prompt carries profile + long-term memory + ask-once rule)
        response = container.assistant.respond(request.message, memory, user_context)

        # 3. Persist with user_id so FUTURE sessions can find this turn
        container.database.save_turn(
            session_id=request.session_id,
            user_query=request.message,
            assistant_response=response,
            user_id=profile.user_id if profile else None,
        )

        # 4. Fold new context into the user's long-term summary every few turns
        container.identity.maybe_update_summary(request.session_id, memory)

        return ChatResponse(session_id=request.session_id,
                            user_message=request.message,
                            assistant_response=response)
    except Exception as e:
        logger.error("/chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
def chat_stream(request: ChatRequest, container: AppContainer = Depends(get_container)):
    """Streaming chat. Identity resolved before streaming; the turn is saved in a finally block."""
    memory = container.session_manager.get_memory(request.session_id)
    try:
        is_first_turn = len(memory.chat_history) == 0
        profile = container.identity.resolve(request.session_id, request.message, memory)
        user_context = container.identity.build_context(profile, is_first_turn)
        user_id = profile.user_id if profile else None
    except Exception as e:
        logger.error("/chat/stream setup failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    def event_generator():
        full_response = ""
        try:
            for chunk in container.assistant.stream(request.message, memory, user_context):
                full_response += chunk
                yield chunk
        except Exception as e:
            logger.error("/chat/stream failed mid-stream: %s", e)
            yield f"\n\n[stream error: {e}]"
        finally:
            if full_response:
                container.database.save_turn(
                    session_id=request.session_id,
                    user_query=request.message,
                    assistant_response=full_response,
                    user_id=user_id,
                )
                container.identity.maybe_update_summary(request.session_id, memory)

    return StreamingResponse(event_generator(), media_type="text/plain")
