import os
import json
import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import TypeDict, Annotated


load_dotenv()

app = FastAPI()

TOOLS = [
    {
        "name" : "read_file",
        "description" : "Read the file and understand it's content",
        "input_schema" : {
            "type" : "object",
            "properties" : {
                "path" : {
                    "type" : "string",
                    "description" : "The file path to read"
                }
            },
            "required" : ["path"]
        }
    },
    {
        "name" : "list_files",
        "description" : "list the file you see",
        "input_schema" : {
            "type" : "object",
            "properties" : {
                "directory" : {
                    "type" : "string",
                    "description" : "The file directory"
                }
            },
            "required" : ["directory"]
        }
    },
    {
        "name" : "write_file",
        "description" : "Get it prepared for write operation",
        "input_schema" : {
            "type" : "object",
            "properties" : {
                "path" : {
                    "type" : "string",
                    "description" : "The file path"
                },
                "content" : {
                    "type" : "string",
                    "description" : "The file content"
                }
            },
            "required" : ["path", "content"]
        }
    }
]
def read_file(path : str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError :
        return f"Error: File not found at {path}"
    except Exception as e:
        return f"Error reading file : {str(e)}"

def list_files(directory : str) -> str:
    try :
        files = os.listdir(directory)
        return "\n".join(files)
    except FileNotFoundError :
        return f"Error : File not found at {directory}"
    except Exception as e:
        return f"Error reading file : {str(e)}"

def write_file(path, content : str) -> str:
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"File written successfully to {path}"
    except FileNotFoundError:
        return f"Error : File not fount at {path}"
    except Exception as e:
        return f"Error writing file : {str(e)}"

def execute_tool(tool_name : str, tool_input: dict) -> str:
    if tool_name == "read_file" :
        read = read_file(tool_input["path"])
        return read
    elif tool_name == "list_files":
        list = list_files(tool_input["directory"])
        return list
    elif tool_name == "write_file":
        write = write_file(tool_input["path"], tool_input["content"])
        return write
    else:
        return f"Unkown tool : {tool_name}"
    


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
    Agentic loop that handles both tool use and text responses.

    1. Send messages to Claude with tools available
    2. If Claude wants to use a tool → execute it, add result to messages, loop back
    3. If Claude is done → stream the final text response to the frontend
    """
    formatted = [{"role": m.role, "content": m.content} for m in messages]

    async with anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) as client:
        while True:
            # Use .create() instead of .stream() so we can inspect the full response
            response = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=formatted,
                tools=TOOLS,
            )

            if response.stop_reason == "tool_use":
                # Claude wants to call a tool — execute it and loop back
                formatted.append({"role": "assistant", "content": response.content})
                for block in response.content:
                    if block.type == "tool_use":
                        result = execute_tool(block.name, block.input)
                        # Notify the frontend which tool is being called
                        yield json.dumps({"type": "tool_call", "tool": block.name, "input": block.input}) + "\n"
                        formatted.append({"role": "user", "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            }
                        ]})
            else:
                # Claude is done — yield the final text response
                for block in response.content:
                    if hasattr(block, "text"):
                        yield json.dumps({"type": "text", "text": block.text}) + "\n"
                break

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
