"""
Interfaces and data models for the feature.
"""

from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List
from typing_extensions import TypedDict

# Models


class Transcript(BaseModel):
    class Chunk(BaseModel):
        text: str
        start: float
        end: float

    text: str = ""
    segments: List[Chunk] = []


class Embeddings(BaseModel):
    class EmbeddingsItem(BaseModel):
        embedding: List[float]

    data: List[EmbeddingsItem] = []


class SearchResult(BaseModel):
    class ScoredChunk(BaseModel):
        score: float
        payload: Transcript.Chunk

    chunks: List[ScoredChunk] = []


# Interfaces (implemented in ./implementations)


class ITranscriber(ABC):
    @abstractmethod
    def get_transcription_from_file(self, file_name: str, file: bytes) -> Transcript:
        raise NotImplementedError


class IEmbedder(ABC):

    @abstractmethod
    def embed(self, inputs: list[str]) -> Embeddings:
        raise NotImplementedError

    @abstractmethod
    def embed_single(self, input: str) -> Embeddings:
        raise NotImplementedError


class IVectors(ABC):
    @abstractmethod
    def create(
        self,
        collection_name: str,
        embeddings: Embeddings,
        segments: List[Transcript.Chunk],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def search(self, collection_name: str, embeddings: Embeddings) -> SearchResult:
        raise NotImplementedError


class IChat(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


"""
Issue:  concrete classes derived from those are injected as FastAPI dependencies
        which silents NotImplementedError (they are not instanciated in the source code, it is delegated to FastAPI)
"""

# Response data types


class TranscriptUploadResponse(TypedDict):
    name: str
    transcript: Transcript


class TranscriptSearchResponse(TypedDict):
    answer: str
