"""
Core logic of the feature: functions using the feature models.
"""

from .models import (
    ITranscriber,
    IEmbedder,
    IVectors,
    IChat,
    TranscriptUploadResponse,
    TranscriptSearchResponse,
)


def create_context_from_audio_file(
    name: str,
    filename: str,
    file: bytes,
    transcriber: ITranscriber,
    embedder: IEmbedder,
    vectors: IVectors,
) -> TranscriptUploadResponse:

    transcription_response = transcriber.get_transcription_from_file(
        file_name=filename,
        file=file,
    )
    embeddings_ctx = embedder.embed([s.text for s in transcription_response.segments])

    name = vectors.create(
        collection_name=name,
        embeddings=embeddings_ctx,
        segments=transcription_response.segments,
    )

    return {"transcript": transcription_response, "name": name}


def generate_query_with_context(
    q: str,
    name: str,
    embedder: IEmbedder,
    vectors: IVectors,
    chat: IChat,
) -> TranscriptSearchResponse:
    embeddings_prompt = embedder.embed_single(q)
    result = vectors.search(name, embeddings_prompt)

    formated_ctx = {
        "chunks": [
            {"text": chunk.payload.text, "time": chunk.payload.start}
            for chunk in result.chunks
        ]
    }
    prompt = make_prompt(ctx=str(formated_ctx), q=q)
    chat_response = chat.complete(prompt=prompt)
    return {"answer": chat_response}


def make_prompt(ctx: str, q: str):
    return f"""
    You are provided with a context coming from an audio transcript.
    Each "chunk" of the list represent a relevant part for you to use as context.
    In each "chunk", there is a "time" timestamp in seconds corresponding to the "text" that you should use to situate your quotes in time.

    CONTEXT:
    ---------------------
    {ctx}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {q}
    Answer:
    """
