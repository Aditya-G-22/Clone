import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

load_dotenv()

app = FastAPI()

# Allow the frontend (running on localhost:5173) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


SYSTEM_PROMPT = """You are an expert coding assistant embedded in a code editor, like Cursor.
Help the user write, understand, debug, and improve their code.
Be concise. Format code with proper markdown code blocks."""


async def stream_chat(messages: list[Message]):
    """
    Async generator that streams Claude's response token by token.

    FastAPI requires an async generator for streaming to work correctly.
    Each chunk is yielded as newline-delimited JSON (NDJSON) so the
    frontend can parse each piece as it arrives.
    """
    formatted = [{"role": m.role, "content": m.content} for m in messages]

    # We use the async client for non-blocking streaming in FastAPI
    async with anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) as async_client:
        async with async_client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=formatted,
        ) as stream:
            async for text in stream.text_stream:
                yield json.dumps({"type": "text", "text": text}) + "\n"

    yield json.dumps({"type": "done"}) + "\n"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    """
    The main chat endpoint. Returns a streaming response.

    StreamingResponse + a generator function = real-time word-by-word output.
    The media type "text/event-stream" is the standard for server-sent events.
    """
    return StreamingResponse(
        stream_chat(request.messages),
        media_type="text/event-stream",
    )
