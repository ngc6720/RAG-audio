from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from functools import lru_cache


class _Variables(BaseSettings):
    # Variables
    environment: str = "development"
    docker_compose: bool = False
    # Secrets
    secret_mistral_api_key: str = "./secrets/mistral_api_key.txt"
    # Put each secret in its own .txt file in ./secrets/ directory.
    # Then add its path (relative to the root of the project) in this file (here and in class _Secrets),
    # and to docker-compose.yml aswell so it works with containers
    # /!\ This is dependent on this file and docker-compose.yaml locations, so avoid moving those files.


# Get the value from the file, in Docker files or in local folder depending on the docker_compose variable (which is switched to true by docker-compose)
def _get_secret(settings: _Variables, secret_path: str, ROOT_PATH: Path) -> str:
    with open(
        f"{'/run/secrets/' if settings.docker_compose else ROOT_PATH }/{secret_path}",
        "r",
    ) as file:
        return file.read().strip()


class _Secrets:
    def __init__(self, settings: _Variables, ROOT_PATH: Path):
        self.mistral_api_key = _get_secret(
            settings, settings.secret_mistral_api_key, ROOT_PATH=ROOT_PATH
        )


class _Config:
    def __init__(self):
        self.ROOT_PATH = Path(__file__).parent / "../"
        self.variables = _Variables()
        self.secrets = _Secrets(self.variables, ROOT_PATH=self.ROOT_PATH)


@lru_cache
def Config():
    return _Config()
