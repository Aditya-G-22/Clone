import os
import anthropic
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Load env variables first so client can read the API key
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list      # conversation history in Anthropic format
    response: any       # Claude's latest response object

# ── Tools Definition ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files in a given directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The directory path to list files from"
                }
            },
            "required": ["directory"]
        }
    },
    {
        "name": "write_file",
        "description": "Write or overwrite a file with the given content",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    }
]

# ── Tool Functions ────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def list_files(directory: str) -> str:
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except FileNotFoundError:
        return f"Error: Directory not found at {directory}"
    except Exception as e:
        return f"Error listing files: {str(e)}"

def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"File written successfully to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "read_file":
        return read_file(tool_input["path"])
    elif tool_name == "list_files":
        return list_files(tool_input["directory"])
    elif tool_name == "write_file":
        return write_file(tool_input["path"], tool_input["content"])
    else:
        return f"Unknown tool: {tool_name}"

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert coding assistant embedded in a code editor, like Cursor.
Help the user write, understand, debug, and improve their code.
Be concise. Format code with proper markdown code blocks."""

# ── LangGraph Nodes ───────────────────────────────────────────────────────────

def call_claude_node(state: AgentState):
    """Node 1 — Call Claude with current messages and tools."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=state["messages"],
        tools=TOOLS,
    )
    return {"response": response}

def execute_tools_node(state: AgentState):
    """Node 2 — Execute whatever tool Claude requested and add result to messages."""
    response = state["response"]
    messages = list(state["messages"])

    messages.append({"role": "assistant", "content": response.content})

    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            messages.append({"role": "user", "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                }
            ]})

    return {"messages": messages}

def should_continue(state: AgentState) -> str:
    """Router — decide whether to execute tools or end the loop."""
    if state["response"].stop_reason == "tool_use":
        return "execute_tools"
    return END

# ── Build & Compile the Graph ─────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("call_claude", call_claude_node)
    graph.add_node("execute_tools", execute_tools_node)

    graph.set_entry_point("call_claude")
    graph.add_conditional_edges("call_claude", should_continue)
    graph.add_edge("execute_tools", "call_claude")

    return graph.compile()

agent = build_graph()

if __name__ == "__main__":
    while True:
        user_input = input("you : ")
        if user_input == "quit":
            break
        else:
            result = agent.invoke({
                "messages": [{"role": "user", "content": user_input}],
                "response": None
            })
        
            final_response = result["response"]
            for block in final_response.content:
                if hasattr(block, "text"):
                    print("Claude:", block.text)