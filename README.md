# RAG-audio

This project is using FastAPI, Mistral AI and Qdrant to provide an API for simple RAG from audio. It allows to post an audio file (that contains speech) and interact with its content via natural language. It is a project made to document learning and experimenting with the tools.

## Installation

```sh
git clone https://github.com/ngc6720/rag-audio.git
cd rag-audio
```

This app uses Mistral AI, you will need an API key to use it. To create a key, connect to your Mistral account, got to [Mistral's settings - API Keys](https://admin.mistral.ai/organization/api-keys) and click on "Create new key". It is recommended to select an expiration date for the key.

Create a folder named "secrets" at the root of the project and, inside, create a "mistral_api_key.txt" file that contains your Mistral public API key.
This command does it all in one go (replace YOUR_API_KEY with your Mistral key):

```sh
mkdir -p secrets && echo YOUR_API_KEY > secrets/mistral_api_key.txt
```

### A) With docker

The simplest way, if you have Docker installed (requires Docker Compose v2.23.1 or above):

```sh
docker compose up -d
```

You can visit http://localhost:6333/dashboard to visualize vectors.

### B) With a virtual environment

/!\ Only for the FastAPI app : Docker would still be needed for any other containerized service, i.e Qdrant for the vectors database.
In that case it will use an in-memory version of the database instead, and it will not be persistant.

#### Create virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate
```

#### Install dependencies

```sh
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Start server dev

```sh
fastapi dev app/main.py --host 0.0.0.0 --port 8000
```

#### Start server prod

```sh
fastapi run app/main.py --host 0.0.0.0 --port 80
```

## Usage

### Swagger UI

With your browser, visit http://localhost:8000/docs to visualise and try endpoints.

Step 1: provide an audio file with a name (create context from audio).
Step 2: provide a prompt and the name of the context to link to your question (ask in context).

You can use the Swagger UI /docs dashboard to do these steps.

### Examples with cURL

Here are examples with cURL commands instead, with the Obama speech from Mistral docs examples:

#### Create context from audio

```sh
curl -X POST -F "file=@test_media/obama.mp3" http://localhost:8000/rag/upload?&name=rag_ctx_collection
```

#### Ask in context

prompt: Should we be hopeful?

```sh
curl http://localhost:8000/rag/search?&q=Should%20we%20be%20hopeful%3F&name=rag_ctx_collection
```

prompt: Is the topic of democracy discussed?

```sh
curl http://localhost:8000/rag/search?&q=Is%20the%20topic%20of%20democracy%20discussed%3F&name=rag_ctx_collection
```

Note: the query parameter "name" is used to identify the context when calling the vector database.

## Results

With ./test_media/obama.mp3, from Mistral Audio & Transcription docs examples, using the above commands:

### Prompt 1

(embeddings_prompt_1.json)

**prompt:**

_Should we be hopeful?_

**answer:**

Based on the provided context, the answer to "Should we be hopeful?" is a resounding **yes**. The speaker expresses optimism and highlights numerous reasons for hope, including:

- The resilience and hope of the American people (39.6s).
- The progress seen in communities like Joplin rebuilding after disaster and Boston's response to terrorism (57.1s).
- Scientific advancements, such as helping a paralyzed man regain his sense of touch and wounded warriors walk again (78.5s).
- The inspiration drawn from young graduates, military officers, and even children reminding us of our obligations to care for others (66.5s, 100.1s).
- The collective effort in self-government, which, though challenging, leads to positive change (111.6s).

The speaker emphasizes that these achievements are possible when people come together, suggesting that hope is justified when we work toward shared goals.

### Prompt 2

(embeddings\*prompt_2.json)

**prompt:**

_Is the topic of democracy discussed?_

**answer:**

Yes, the topic of democracy is discussed in the provided context. Several chunks explicitly mention democracy, such as:

- "It falls on each of us to be guardians of our democracy." (time: 165.2)
- "But we can't take our democracy for granted." (time: 120.3)
- "That's what's possible when we come together in the slow, hard, sometimes frustrating, but always vital work of self-government." (time: 111.6)
- "Our success depends on our participation, regardless of which way the pendulum of power swings." (time: 156.9)

These quotes highlight the importance of active citizenship and collective effort in sustaining democracy.
