# =============================================================================
# CURSOR CLONE — CLI Coding Agent
# =============================================================================
# A command-line AI coding assistant inspired by Cursor and Claude Code.
# You type a task in the terminal, the agent reasons about it, uses tools
# if needed, and gives you a final answer.
#
# Stack:
#   - LangGraph  → manages the agent loop as a state graph
#   - LangChain  → provides the tool interface and model abstraction
#   - Groq/Anthropic → the actual LLM doing the reasoning (swappable)
#   - Rich       → makes the terminal output look good
# =============================================================================

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

# ──────────────────────────────── ENVIRONMENT ───────────────────────────────────────────────────────────────
# Must run before anything that needs API keys.

load_dotenv()

# ──────────────────────────────── RICH CONSOLE ──────────────────────────────────────────────────────────────
# Single Console instance used everywhere for styled output.
# Think of it as a smarter version of print().

console = Console()

# ────────────────────────────────────── STATE ─────────────────────────────────────────────────────────────────────
# AgentState is the memory that flows through every node in the graph.
# `messages` holds the full conversation history.
# `add_messages` is a reducer — it appends new messages instead of replacing.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ────────────────────────────────────── TOOLS ─────────────────────────────────────────────────────────────────────
# Tools give the LLM the ability to act on the real world.
# The LLM cannot touch your filesystem — it requests a tool call,
# and our Python functions execute it.
#
# Rules:
#   - Each function must have a docstring (LangChain uses it as the description)
#   - Always return a string (the result goes back to the LLM as text)
#   - Always use try/except (a bad path should not crash the agent)

BLOCKED_FILES = [".env", ".env.example"]   # files the agent is never allowed to read

@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    if os.path.basename(path) in BLOCKED_FILES:
        return f"Error: Access to '{path}' is blocked for security reasons."
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at '{path}'"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def list_files(directory: str) -> str:
    """List all files and folders in a given directory."""
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except FileNotFoundError:
        return f"Error: Directory not found at '{directory}'"
    except Exception as e:
        return f"Error listing files: {str(e)}"

@tool
def write_file(path: str, content: str) -> str:
    """Write or overwrite a file with the given content."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"File written successfully to '{path}'"
    except Exception as e:
        return f"Error writing file: {str(e)}"

tools = [read_file, list_files, write_file]

# ────────────────────────── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# Hidden instructions sent to the LLM on every request.
# Shapes its personality and behavior. The user never sees this.

SYSTEM_PROMPT = """You are an expert coding assistant like Cursor.
Help the user write, understand, debug, and improve their code.
Only use tools (read_file, list_files, write_file) when the user explicitly asks you to read, list or write a file.
For general questions, answer directly without using tools.
Be concise. Format code with proper markdown code blocks."""

# ────────────────────────── GRAPH NODES ───────────────────────────────────────────────────────────────
# A node is a function that receives state, does something, and returns
# only the fields that changed. LangGraph merges the changes into state.

def call_claude_node(state: AgentState):
    """Node 1 — Send messages to the LLM and get a response."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def execute_tools_node(state: AgentState):
    """Node 2 — Execute the tool(s) the LLM requested."""
    last_message = state["messages"][-1]
    tool_map = {t.name: t for t in tools}
    results = []
    for tool_call in last_message.tool_calls:
        # Show the user which tool is being called
        console.print(f"  [dim]⚙ using tool:[/dim] [yellow]{tool_call['name']}[/yellow] [dim]{tool_call['args']}[/dim]")
        result = tool_map[tool_call["name"]].invoke(tool_call["args"])
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    return {"messages": results}

def should_continue(state: AgentState) -> str:
    """Router — if the LLM made tool calls, execute them. Otherwise end."""
    if state["messages"][-1].tool_calls:
        return "execute_tools"
    return END

# ──────────────────────────────── GRAPH ─────────────────────────────────────────────────────────────────────
# Wires the nodes together into a loop:
#
#   [START] → call_claude → tool_use? → execute_tools ─┐
#                        → end_turn?  → [END]           │
#                  ↑___________________________________ ─┘

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("call_claude", call_claude_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.set_entry_point("call_claude")
    graph.add_conditional_edges("call_claude", should_continue)
    graph.add_edge("execute_tools", "call_claude")
    return graph.compile()

agent = build_graph()

# ────────────────────────────────────── CLI ───────────────────────────────────────────────────────────────────────
# Entry point when running: python main.py
# Shows a banner, asks the user to pick a model, then starts a chat loop.
# Each message is independent — no memory between turns.

if __name__ == "__main__":

    # Banner
    console.print(Panel.fit(
        "[bold cyan]Cursor Clone[/bold cyan] — CLI Coding Agent\n"
        "[dim]Powered by LangGraph + LangChain[/dim]",
        border_style="cyan"
    ))

    # Model selection
    console.print("\n[bold]Select a model:[/bold]")
    console.print("  [cyan]1.[/cyan] Llama 3.3 70b  [dim](free  — Groq)[/dim]")
    console.print("  [cyan]2.[/cyan] Claude Sonnet  [dim](paid  — Anthropic)[/dim]")

    choice = console.input("\n[bold cyan]Enter 1 or 2:[/bold cyan] ").strip()

    if choice == "1":
        MODEL_NAME = "llama-3.3-70b-versatile"
        model = ChatGroq(model=MODEL_NAME)
    elif choice == "2":
        MODEL_NAME = "claude-sonnet-4-6"
        model = ChatAnthropic(model=MODEL_NAME)
    else:
        console.print("[yellow]Invalid choice — defaulting to Llama[/yellow]")
        MODEL_NAME = "llama-3.3-70b-versatile"
        model = ChatGroq(model=MODEL_NAME)

    # Bind tools AFTER model is selected so the right model gets them
    model_with_tools = model.bind_tools(tools)

    console.print(f"\n[dim]Model  :[/dim] [green]{MODEL_NAME}[/green]")
    console.print("[dim]Type 'quit' to exit[/dim]\n")
    console.print(Rule(style="dim"))

    # Chat loop
    while True:
        user_input = console.input("\n[bold green]you:[/bold green] ").strip()

        if user_input.lower() == "quit":
            console.print(Panel.fit("[dim]Goodbye![/dim]", border_style="dim"))
            break

        if not user_input:
            continue

        # Spinner while the agent is thinking
        try:
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                result = agent.invoke({
                    "messages": [HumanMessage(content=user_input)]
                })

            # Render the response as markdown inside a panel
            response_text = result["messages"][-1].content
            console.print(Panel(
                Markdown(response_text),
                title="[bold cyan]Assistant[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            ))

        except Exception as e:
            console.print(Panel(
                f"[red]Error:[/red] The model failed to respond correctly.\n"
                f"Try rephrasing or switch to Claude.\n\n[dim]{str(e)[:150]}[/dim]",
                title="[red]Error[/red]",
                border_style="red"
            ))
