import json
from typing import Annotated
from fastapi import Depends
from mistralai import Mistral
from ....config import Config
from ..models import IChat
from ..infra import get_client_mistral


class Chat(IChat):
    def __init__(self, client: Annotated[Mistral, Depends(get_client_mistral)]):
        self.client = client

    def complete(self, prompt):
        chat_response = self.client.chat.complete(
            model="mistral-small-latest", messages=[{"role": "user", "content": prompt}]
        )
        return str(chat_response.choices[0].message.content)


class ChatMock(IChat):
    def complete(self, prompt):
        with open(f"{Config().ROOT_PATH}/test_media/answer_1.json") as f:
            result = json.load(f)
            return result["answer"]
