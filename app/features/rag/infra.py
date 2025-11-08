from functools import lru_cache
from mistralai import Mistral
from qdrant_client import QdrantClient
from ...config import Config


@lru_cache
def get_client_mistral():
    return Mistral(api_key=Config().secrets.mistral_api_key)


@lru_cache
def get_client_qdrant_memory():
    return QdrantClient(":memory:")


@lru_cache
def get_client_qdrant_docker():
    return QdrantClient(host="host.docker.internal", port=6333)
