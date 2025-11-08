import json
import requests
from typing import Annotated
from fastapi import Depends
from mistralai import Mistral, models as mistral_models
from ....config import Config
from ..infra import get_client_mistral
from ..models import ITranscriber, Transcript


class TranscriberMock(ITranscriber):
    def get_transcription_from_file(
        self,
        file_name: str,
        file: bytes,
    ):
        with open(f"{Config().ROOT_PATH}/test_media/transcript.json") as f:
            result = json.load(f)["transcription"]
            return Transcript.model_validate(result)


class TranscriberApi(ITranscriber):
    def get_transcription_from_file(
        self,
        file_name: str,
        file: bytes,
    ):
        url = "https://api.mistral.ai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {Config().secrets.mistral_api_key}",
        }

        files = {"file": ("conversation_sesameai.wav", file)}
        data = {
            "model": "voxtral-mini-latest",
            "timestamp_granularities": ["segment"],
        }
        response = requests.post(url, files=files, data=data, headers=headers)
        return Transcript.model_validate(response.json())


# Issue: does not give timestamps but it works in javascript sdk and also by using directly the api (sending file as bytes too)
class TranscriberSdk(ITranscriber):
    def __init__(self, client: Annotated[Mistral, Depends(get_client_mistral)]):
        self.client = client

    def get_transcription_from_file(self, file_name, file):

        result = self.client.audio.transcriptions.complete(
            model="voxtral-mini-latest",
            timestamp_granularities=[
                "segment"
            ],  #  Fails to apply timestamps with file as Dict
            file={
                "file_name": file_name,
                "content": file,
            },
            # file=mistral_models.File(file_name=file_name, content=file), # No segments returned using File model either
            # file_url="", # But segments are returned when using an url instead
        )
        print(result)  # Segments list is empty.

        return Transcript.model_validate(result, from_attributes=True)
