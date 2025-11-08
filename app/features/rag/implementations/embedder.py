import json
from pydantic import BaseModel
from typing import Annotated
from fastapi import Depends
from mistralai import Mistral
from ....config import Config
from ..infra import get_client_mistral
from ..models import IEmbedder, Embeddings


class EmbedderSdk(IEmbedder):
    def __init__(self, client: Annotated[Mistral, Depends(get_client_mistral)]):
        self.client = client

    def embed(self, inputs):
        embeddings_batch_response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=inputs,
        )
        return Embeddings.model_validate(
            embeddings_batch_response, from_attributes=True
        )

    def embed_single(self, input):
        return self.embed([input])


class EmbedderMock(IEmbedder):
    def embed(self, inputs):
        with open(f"{Config().ROOT_PATH}/test_media/embeddings_ctx.json") as f:
            result = json.load(f)
            return Embeddings.model_validate(result)

    def embed_single(self, input):
        with open(f"{Config().ROOT_PATH}/test_media/embeddings_prompt_1.json") as f:
            result = json.load(f)
            return Embeddings.model_validate(result)


# Only used to make some mock data from received results
def write_to_json_file(data: BaseModel, path: str):
    with open(path, "w") as f:
        json.dump(data.model_dump(), f)
