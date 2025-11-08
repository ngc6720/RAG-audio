import logging
from typing import Dict, Any
from typing_extensions import TypedDict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from .config import Config
from .features.rag import router as transcript_router

from .features.rag.implementations.transcriber import TranscriberApi, TranscriberMock
from .features.rag.implementations.embedder import EmbedderSdk, EmbedderMock
from .features.rag.implementations.chat import Chat, ChatMock
from .features.rag.infra import get_client_qdrant_docker, get_client_qdrant_memory

# App

APP_NAME = "RAG-audio"

app = FastAPI(title=APP_NAME)
logger = logging.getLogger(__name__)

# Dependencies overrides

if Config().variables.docker_compose == True:
    app.dependency_overrides[get_client_qdrant_memory] = get_client_qdrant_docker

"""
Uncomment these overrides below to use mock implementations (could also be set from
specific variables outside the source code, or used for testing).
"""
# app.dependency_overrides[TranscriberApi] = TranscriberMock
# app.dependency_overrides[EmbedderSdk] = EmbedderMock
# app.dependency_overrides[Chat] = ChatMock

# Routes


class InfoResponse(TypedDict):
    ROOT_PATH: str
    variables: Dict[str, Any]
    secrets: Dict[str, None]


class RootResponse(TypedDict):
    app: str
    description: str


@app.get("/")
def read_root() -> RootResponse:
    return {
        "app": APP_NAME,
        "description": "This project is using FastAPI, Mistral AI and Qdrant to provide an API for simple RAG from audio. It allows to post an audio file (that contains speech) and interact with its content via natural language. It is a project made for learning and experimenting with the tools.",
    }


@app.get("/info", description="Config informations (secrets values are hidden).")
def info() -> InfoResponse:
    return {
        "ROOT_PATH": str(Config().ROOT_PATH),
        "variables": Config().variables.model_dump(),
        "secrets": dict.fromkeys(list(vars(Config().secrets).keys())),
    }


app.include_router(transcript_router.router)


# Errors


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"**HTTP error:** {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "path": request.url.path,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"**Unexpected error:** {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "path": request.url.path,
        },
    )
