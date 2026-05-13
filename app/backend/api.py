from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from app.core.ai_agent import get_response_from_ai_agents
from app.config.settings import settings
from app.common.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="MULTI AI AGENT")


class RequestState(BaseModel):
    model_name: str
    system_prompt: str
    messages: List[str]
    allow_search: bool


@app.post(
    "/chat",
    responses={
        400: {"description": "Invalid model name"},
        500: {"description": "Internal Server Error"},
    },
)
def chat_endpoint(request: RequestState):

    logger.info(f"Received request for model: {request.model_name}")

    if request.model_name not in settings.ALLOWED_MODEL_NAMES:
        logger.warning("Invalid model name received")

        raise HTTPException(
            status_code=400,
            detail="Invalid model name"
        )

    try:
        response = get_response_from_ai_agents(
            request.model_name,
            request.messages,
            request.allow_search,
            request.system_prompt
        )

        logger.info(
            f"Successfully got response from AI Agent: {request.model_name}"
        )

        return {
            "response": response
        }

    except Exception as e:

        logger.exception(
            f"Error occurred during response generation: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to get AI response"
        )


