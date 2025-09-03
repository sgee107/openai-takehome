# Takehome Application

Application made for OpenAI takehome test

## Project Setup

Get up and running with the application in just a few steps.

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/) and Docker Compose

### Initial Setup

Use this initial setup for running test

#### 1. Copy .env.example to .env

```bash
cp .env.example .env
```

#### 2. Add in your OPENAI_API_KEY to the .env file
OPENAI_API_KEY=sk-your-key-here

#### 3. Run docker compose up

```bash
docker compose up
```

#### 4. You are free to run tests!

Use curl to hit the api or navigate to localhost:3000 to play with the frontend

## Sample Usage

### Quick Test Questions

Test the search functionality with these sample questions based on the dataset:

#### Simple Search Queries
1. "Show me blue socks for men"
2. "I need palazzo pants for women"
3. "Find girls' dresses size 9-10"
4. "Looking for compression sleeves"
5. "Show me vintage style dresses"

#### Color & Style Searches
6. "I want navy blue clothing"
7. "Find burgundy colored items"
8. "Show me tie dye sweatshirts"
9. "Looking for gold jewelry"
10. "Find silver accessories"

### Testing via Terminal

To test these questions via terminal, you can use curl commands:

```bash
# Test a simple search query
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me blue socks for men"}'

# Test a complex search query
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Find high-rated compression gear"}'

# Test with specific parameters
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Looking for navy blue clothing", "limit": 5}'
```

### Testing via API Documentation

Alternatively, visit `http://localhost:8000/docs` and use the interactive API documentation to test these queries directly in your browser.

## Developer Setup

#### 1. Start Infrastructure Services

Start PostgreSQL and MinIO services:

```bash
docker compose -f docker-compose-dev.yml up -d
```

This will start:
- **PostgreSQL** (port 5432) - Database with pgvector extension
- **MinIO** (ports 9000/9001) - S3-compatible object storage

#### 2. Install Dependencies

```bash
uv sync
```

#### 3. Run the Backend

Start the FastAPI application:

```bash
uv run fastapi dev app/app.py
```

The API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Services Access

- **MinIO Console**: http://localhost:9001 (admin/minioadmin)
- **API Documentation**: http://localhost:8000/docs

### Environment Configuration

Copy `.env.example` to `.env` and adjust settings as needed:

```bash
cp .env.example .env
```

**NOTE**: Must include OPENAI_API_KEY, only non defaulted variable
OPENAI_API_KEY=sk-your-key-here

### Testing

Run the test suite:

```bash
uv run pytest
```

### Stopping Services

```bash
docker compose -f docker-compose-dev.yml down
```





## Key Design decisions and trade-offs

Given the time and overall complexity with creating a production solution, I made the following decisions.

1) I was going to test different embeddings strategies just to show HOW it could be done. I was always intending on adding in information extracted from the image as well.
2) Search is usually defined by an overall success function like a user buying the product. I choosed to just make this somewhat simple for the sake of the exercise and include a linear combination of the embedding cosine similarity search and the average rating.
3) I added in MLFlow as a developer helper which could be used to trace LLM calls and track prompt versioning, it largely was not in this case.
4) The testing is far less built out than I normally would have it. In particular, I would really want to beef up the test cases to red team the inputs and make sure the system prompts are working as intended.
5) I decided to use posgres w/ pgvector as the database just for ease of use, if not pgvector I would have used Chroma
6) The agentic setup for the chat calls could be much much more complex. I just wanted to show some level of interpretation of user intent with the question to map to underlying tool usage. 
7) The chat is solely one at a time just to show the search implementation working. Adding in chat conversations would change the prompt and require some more testing to validate that the system was operating as intended. It could also bring into play more session specifics about the user that could inform the search downstream.
8) I explicitly DID NOT add the ratings into the overall embeddings created, it felt like this was the wrong idea and to keep this separate and to use in a re-ranking prior to return to the user. 
9) I made a lot of values configurable so that we could make changes at runtime later, the idea for search in particular is that you'd want to run a lot of experiments and it would be important to be able to change at run time.
10) In an ideal world I'd continue to beef up the data pipeline. Think the abstractions there could use a lot of work to pipeline and clean up.
11) I created a frontend just for kicks. Figured with NextJS it would be nice to have a skeleton rendered while waiting for the search agent to come to a decision.
12) Streaming wasn't allowed without adding a credit card to the candidate project, so there's no streaming of information as it comes in.
13) I just made a pretty straight forward agentic workflow where I had one master tool being called for now. This could be split up, in particular given the intent at the top, we could offer different types of searches or limits based on the query. We could also have the agent be able to conversate and return multiple types of outputs by using different tools down the line.


**Note**: I did use Cline + OpenAI & Codex for implementation, I have a few implementation plans I saved in the `/docs` for your records.

I've also included some statistics from running the embedding experiments through a few times.
