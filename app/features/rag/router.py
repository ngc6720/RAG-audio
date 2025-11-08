from fastapi import APIRouter, UploadFile, Depends, HTTPException
from typing import Annotated

from .models import (
    ITranscriber,
    IEmbedder,
    IVectors,
    IChat,
    TranscriptUploadResponse,
    TranscriptSearchResponse,
)
from . import service
from .implementations.transcriber import TranscriberApi
from .implementations.embedder import EmbedderSdk
from .implementations.vectors import Vectors
from .implementations.chat import Chat

embedder_dep = Annotated[IEmbedder, Depends(EmbedderSdk)]
vectors_dep = Annotated[IVectors, Depends(Vectors)]

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/upload",
    status_code=201,
    description="Upload an audio file to generate a context to retrieve from and give it a name.",
)
async def create_context_from_audio_file_handler(
    name: str,
    file: UploadFile,
    transcriber: Annotated[ITranscriber, Depends(TranscriberApi)],
    embedder: embedder_dep,
    vectors: vectors_dep,
) -> TranscriptUploadResponse:
    try:
        return service.create_context_from_audio_file(
            name=name,
            filename=file.filename or "untitled",
            file=await file.read(),
            transcriber=transcriber,
            embedder=embedder,
            vectors=vectors,
        )
    except:
        raise HTTPException(status_code=404)


@router.get(
    "/search",
    description="Get chat completion augmented with retrieval from your previous upload (located by its name).",
)
def generate_query_with_context_handler(
    q: str,
    name: str,
    embedder: embedder_dep,
    vectors: vectors_dep,
    chat: Annotated[IChat, Depends(Chat)],
) -> TranscriptSearchResponse:
    try:
        return service.generate_query_with_context(q, name, embedder, vectors, chat)
    except:
        raise HTTPException(status_code=404)
