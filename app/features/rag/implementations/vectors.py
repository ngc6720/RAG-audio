from typing import Annotated
from fastapi import Depends
from qdrant_client import QdrantClient, models
from ..infra import get_client_qdrant_memory
from ..models import IVectors, SearchResult


class Vectors(IVectors):
    def __init__(
        self, client: Annotated[QdrantClient, Depends(get_client_qdrant_memory)]
    ):
        self.client = client

    def create(self, collection_name, embeddings, segments):
        points = [
            models.PointStruct(
                id=idx,
                vector=response.embedding or [],
                payload=segment.model_dump(),
            )
            for idx, (response, segment) in enumerate(zip(embeddings.data, segments))
        ]

        if self.client.collection_exists(collection_name=collection_name):
            self.client.delete_collection(collection_name=collection_name)

        self.client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
            ),
        )
        self.client.upsert(collection_name, points)

        return collection_name

    def search(self, collection_name, embeddings):
        data = self.client.search(
            collection_name=collection_name,
            query_vector=embeddings.data[0].embedding or [],
            with_payload=True,
        )
        return SearchResult.model_validate({"chunks": data}, from_attributes=True)
