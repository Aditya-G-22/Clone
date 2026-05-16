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
import time
import subprocess
import requests
from bs4 import BeautifulSoup
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver 
from langgraph.types import interrupt, Command
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, trim_messages, RemoveMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

# load API keys from .env file
load_dotenv()

# one console used everywhere for pretty output
console = Console()

# ────────────────────────────────────── STATE ─────────────────────────────────────────────────────────────────────
# this is basically the memory of the agent — all messages live here
# add_messages makes sure new messages are appended, not replaced

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iterations: int  # safety counter — stops infinite tool loops
    summary : str # compressed memory of old conversations
    retries : int # keeps the counter as to how many times agent has retried after an error

# ────────────────────────────────────── TOOLS ─────────────────────────────────────────────────────────────────────
# tools are the hands of the agent — it can't do anything without them
# the LLM asks for a tool, we run it, and send the result back

# these files are never readable — security measure
BLOCKED_FILES = [".env", ".env.example"]

@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    if os.path.basename(path) in BLOCKED_FILES:
        return f"Error: reading '{path}' is not allowed."
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: no file found at '{path}'"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def list_files(directory: str) -> str:
    """List all files and folders in a given directory."""
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except FileNotFoundError:
        return f"Error: no directory found at '{directory}'"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def write_file(path: str, content: str) -> str:
    """Write or overwrite a file with the given content."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Done — file saved to '{path}'"
    except Exception as e:
        return f"Error: {str(e)}"
    
@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing a specific string with a new string.
    Use this instead of write_file when making small changes to existing files."""
    try:
        with open(path, "r") as f:
            content = f.read()
        
        if old_string not in content:
            return f"Error: could not find the specific text in '{path}'. No chages made"
        
        if content.count(old_string) > 1:
            return f"Error: found multiple matches for the specified text in '{path}'. Make it more specific."
        
        new_content = content.replace(old_string, new_string, 1)
        with open(path, "w") as f:
            f.write(new_content)
        
        return f"Done — '{path}' updated successfully."
    except FileNotFoundError:
        return f"Error: no file found at '{path}'"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def delete_file(path: str) -> str:
    """Delete a file at the given path."""
    try:
        os.remove(path)
        return f"Done — '{path}' has been deleted."
    except FileNotFoundError:
        return f"Error: no file found at '{path}'"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def run_command(command: str) -> str:
    """Run a terminal command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # replace unreadable chars instead of crashing
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip() or "Command ran successfully (no output)."
        else:
            return f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: command took too long (30s timeout)"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def search_web(query: str) -> str:
    """Search the web for a topic and return top results with titles, URLs and summaries."""
    try:
        tavily = TavilySearch(max_results=5)
        results = tavily.invoke({"query": query})
        if isinstance(results, list):
            formatted = []
            for r in results:
                formatted.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('url', 'N/A')}\n"
                    f"Content: {r.get('content', 'N/A')}"
                )
            return "\n---\n".join(formatted)
        return str(results)
    except Exception as e:
        return f"Error searching web: {str(e)}"

@tool
def read_url(url: str) -> str:
    """Fetch and read the text content of a webpage or article."""
    try:
        # pretend to be a browser so websites don't block us
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # remove stuff we don't need like nav bars, scripts, footers
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # cut it off at 5000 chars so we don't overload the LLM context
        return text[:5000]
    except Exception as e:
        return f"Error reading URL: {str(e)}"

# all tools in one list — execute_tools_node uses this to find the right function
tools = [read_file, list_files, write_file, edit_file, delete_file, run_command, search_web, read_url]

# ────────────────────────── SYSTEM PROMPTS ─────────────────────────────────────────────────────────────
# each agent gets its own personality via a different system prompt
# the user never sees these — they shape how the LLM behaves

CODING_PROMPT = """You are an expert coding assistant like Cursor.
Help the user write, understand, debug, and improve their code.
Only use tools when the task requires it.
For general questions, answer directly without using tools.
Be concise. Format code with proper markdown code blocks.
If a tool returns an error, analyze the error message, fix your approach, and try again."""

RESEARCH_PROMPT = """You are a thorough research assistant.
When given a topic, search the web and read relevant articles to find accurate information.
Always search first, then read the most relevant URLs for deeper information.
Summarize your findings clearly and mention your sources.
If a tool returns an error, analyze the error message, fix your approach, and try again."""

FILE_PROMPT = """You are a file management assistant.
Help the user read, write, list and delete files on their system.
Always use the appropriate file tool for the task.
Prefer edit_file over write_file when modifying existing files — only use write_file for creating new files.
Be careful with delete operations — confirm what you're doing.
If a tool returns an error, analyze the error message, fix your approach, and try again."""

TERMINAL_PROMPT = """You are a terminal assistant.
Run the commands the user asks for and explain the output.
If a command looks dangerous, warn the user before running it.
If a tool returns an error, analyze the error message, fix your approach, and try again."""

# ────────────────────────── GRAPH NODES ───────────────────────────────────────────────────────────────
# nodes are the workers in the graph — each one does one job
# they receive the current state, do something, and return what changed

# this is a closure — it lets us create a node with a specific model + prompt baked in
# without closures we'd have to use globals, which gets messy
def make_call_model_node(llm, system_prompt=CODING_PROMPT):
    def call_model_node(state: AgentState):
        # trim old messages first, then add system prompt on top
        trimmed = trim_messages(
            state["messages"],
            max_tokens=20,
            strategy="last",
            token_counter=len,
            include_system=True,
        )
        summary = state.get("summary", "")
        if summary:
            system_with_memory = system_prompt + f"\n\nContext from previous conversation:\n{summary}"
        else:
            system_with_memory = system_prompt

        messages = [SystemMessage(content=system_with_memory)] + trimmed
        response = llm.invoke(messages)
        return {"messages": [response]}
    return call_model_node

def execute_tools_node(state: AgentState):
    # the LLM asked for tools — we run them here and send results back
    last_message = state["messages"][-1]
    tool_map = {t.name: t for t in tools}
    results = []

    # these tools can cause damage — always ask the user first
    DANGEROUS_TOOLS = {"delete_file", "run_command"}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name in DANGEROUS_TOOLS:
            # interrupt() pauses the entire graph here and sends this message up to the CLI
            # the graph won't move until agent.invoke(Command(resume=answer)) is called
            answer = interrupt(
                f"Tool: {tool_name}\n"
                f"Args: {tool_args}\n"
                f"Type 'yes' to confirm or anything else to cancel."
            )
            if answer.strip().lower() not in ("yes", "y"):
                console.print(f"  [dim]✗ cancelled:[/dim] [red]{tool_name}[/red]")
                results.append(ToolMessage(
                    content="Action cancelled by user.",
                    tool_call_id=tool_call["id"]
                ))
                continue

        console.print(f"  [dim]⚙ using tool:[/dim] [yellow]{tool_name}[/yellow] [dim]{tool_args}[/dim]")
        result = tool_map[tool_name].invoke(tool_args)
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    has_error = any("Error" in str(r.content) for r in results)
    return {
        "messages": results,
        "iterations": state.get("iterations", 0) + 1,
        "retries": state.get("retries", 0) + (1 if has_error else 0)
    }

def should_continue(state: AgentState) -> str:
    # did the LLM ask for a tool? if yes keep going, if no we're done
    if state["messages"][-1].tool_calls:
        if state.get("iterations", 0) >= 10:
            console.print("  [dim yellow]⚠ max iterations reached — stopping loop[/dim yellow]")
            return END
        if state.get("retries", 0) >= 3:
            console.print("  [dim yellow]⚠ max retries reached — stopping loop[/dim yellow]")
            return END
        return "execute_tools"
    return END

def should_summarize(state: AgentState) -> str:
    if len(state["messages"]) > 10:
        return "summarize"
    return END

def make_summarize_node(llm) :
    def summarize_node(state: AgentState) :
        summary = state.get("summary", "")
        messages = state["messages"]
        if summary:
            prompt = (
                f"This is the existing summary:\n{summary}\n\n"
                f"Extend it with the new messages above in 3-4 sentences:"
            )
        else:
            prompt = "Summarize this conversation in 3-4 sentences, focusing on what was done and any important context:"

        response = llm.invoke(messages + [HumanMessage(content=prompt)])

        # delete everything except the last 4 messages
        delete_messages = [RemoveMessage(id=m.id) for m in messages[:-4]]
        console.print("[dim green]✓ summarizing conversation...[/dim green]")

        return {"summary": response.content, "messages": delete_messages}
    return summarize_node
        
# ────────────────────────── ROUTERS ───────────────────────────────────────────────────────────────────
# routers are just functions that look at the message and decide which agent should handle it
# we use a small fast model for this so it doesn't slow things down

def make_router(llm):
    # top level router — decides between file, code, and research agents
    def router(state: AgentState) -> str:
        last_message = state["messages"][-1].content
        prompt = f"""Classify this user message into one of three categories:
- "file_agent": user wants to read, write, list or delete a LOCAL file or folder (message mentions a filename like main.py, test.txt, or a directory path)
- "research_agent": user wants to search the web, research a topic, read online articles or find information on the internet
- "code_agent": user wants coding help, debugging, explanations, or wants to run a terminal/shell command

IMPORTANT: "read main.py" or "read any_file.ext" = file_agent — it has a filename.
IMPORTANT: "read this article" or "search for X" = research_agent — it's about the internet.
IMPORTANT: "run", "execute", or any shell command (ls, git, pip) = code_agent.

Message: {last_message}

Reply with ONLY "file_agent", "research_agent", or "code_agent". Nothing else."""

        response = llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()

        if "file_agent" in decision:
            console.print("  [dim]→ routing to:[/dim] [magenta]file_agent[/magenta]")
            return "file_agent"
        if "research_agent" in decision:
            console.print("  [dim]→ routing to:[/dim] [magenta]research_agent[/magenta]")
            return "research_agent"
        console.print("  [dim]→ routing to:[/dim] [magenta]code_agent[/magenta]")
        return "code_agent"
    return router

def make_code_router(llm):
    # internal router inside code_agent — decides between terminal and plain coding
    def code_router(state: AgentState) -> str:
        last_message = state["messages"][-1].content
        prompt = f"""Classify this message:
- "terminal_agent": user wants to run a terminal or shell command (like ls, git, pip, python, etc.)
- "call_model": user wants coding help, explanation or debugging (no terminal needed)

Message: {last_message}

Reply with ONLY "terminal_agent" or "call_model". Nothing else."""

        response = llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()

        if "terminal_agent" in decision:
            console.print("  [dim]→ routing to:[/dim] [magenta]terminal_agent[/magenta]")
            return "terminal_agent"
        console.print("  [dim]→ routing to:[/dim] [magenta]call_model[/magenta]")
        return "call_model"
    return code_router

# ──────────────────────────────── SUBGRAPHS ──────────────────────────────────────────────────────────────
# each subgraph is a mini agent with its own tools and loop
# they look like a single node from the outside but are full graphs inside

def build_file_agent(model):
    file_llm = model.bind_tools([read_file, list_files, write_file, edit_file, delete_file])
    graph = StateGraph(AgentState)
    graph.add_node("call_model", make_call_model_node(file_llm, FILE_PROMPT))
    graph.add_node("execute_tools", execute_tools_node)
    graph.set_entry_point("call_model")
    graph.add_edge("execute_tools", "call_model")
    graph.add_conditional_edges("call_model", should_continue)
    return graph.compile()

def build_research_agent(model):
    # research agent gets search + read tools so it can look stuff up
    research_llm = model.bind_tools([search_web, read_url])
    graph = StateGraph(AgentState)
    graph.add_node("call_model", make_call_model_node(research_llm, RESEARCH_PROMPT))
    graph.add_node("execute_tools", execute_tools_node)
    graph.set_entry_point("call_model")
    graph.add_conditional_edges("call_model", should_continue)
    graph.add_edge("execute_tools", "call_model")
    return graph.compile()

def build_terminal_agent(model):
    terminal_llm = model.bind_tools([run_command])
    graph = StateGraph(AgentState)
    graph.add_node("call_model", make_call_model_node(terminal_llm, TERMINAL_PROMPT))
    graph.add_node("execute_tools", execute_tools_node)
    graph.set_entry_point("call_model")
    graph.add_conditional_edges("call_model", should_continue)
    graph.add_edge("execute_tools", "call_model")
    return graph.compile()

def build_code_agent(model, router_model):
    # code agent has its own internal router — it routes to terminal or plain LLM
    graph = StateGraph(AgentState)
    terminal_agent = build_terminal_agent(model)
    graph.add_node("terminal_agent", terminal_agent)
    graph.add_node("call_model", make_call_model_node(model, CODING_PROMPT))
    graph.add_conditional_edges(START, make_code_router(router_model))
    graph.add_edge("terminal_agent", END)
    graph.add_edge("call_model", END)
    return graph.compile()

def build_graph(model, router_model, memory):
    # the parent graph — all agents live here as nodes
    graph = StateGraph(AgentState)
    graph.add_node("file_agent", build_file_agent(model))
    graph.add_node("research_agent", build_research_agent(model))
    graph.add_node("code_agent", build_code_agent(model, router_model))
    graph.add_node("summarize", make_summarize_node(model))
    graph.add_conditional_edges(START, make_router(router_model))
    graph.add_conditional_edges("file_agent", should_summarize)
    graph.add_conditional_edges("research_agent", should_summarize)
    graph.add_conditional_edges("code_agent", should_summarize)
    graph.add_edge("summarize", END)
    return graph.compile(checkpointer=memory)

# ────────────────────────── MODEL SELECTION ───────────────────────────────────────────────────────────────────────
# pulled into a function so we can call it again when user types "switch"

def select_model():
    console.print("\n[bold]Select a model:[/bold]")
    console.print("  [cyan]1.[/cyan] Llama 3.3 70b  [dim](free  — Groq)[/dim]")
    console.print("  [cyan]2.[/cyan] Claude Sonnet  [dim](paid  — Anthropic)[/dim]")

    choice = console.input("\n[bold cyan]Enter 1 or 2:[/bold cyan] ").strip()

    if choice == "1":
        model_name = "llama-3.3-70b-versatile"
        model = ChatGroq(model=model_name)
    elif choice == "2":
        model_name = "claude-sonnet-4-6"
        model = ChatAnthropic(model=model_name)
    else:
        console.print("[yellow]Invalid choice — defaulting to Llama[/yellow]")
        model_name = "llama-3.3-70b-versatile"
        model = ChatGroq(model=model_name)

    return model, model_name

# ────────────────────────────────────── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

  with SqliteSaver.from_conn_string("memory.db") as memory:

    console.print(Panel.fit(
        "[bold cyan]Cursor Clone[/bold cyan] — CLI Coding Agent\n"
        "[dim]Powered by LangGraph + LangChain[/dim]",
        border_style="cyan"
    ))

    # pick the main model
    model, MODEL_NAME = select_model()

    # small fast model just for routing — saves time on every message
    router_model = ChatGroq(model="llama-3.1-8b-instant")

    # build the full agent graph — memory passed in from the with block above
    agent = build_graph(model, router_model, memory)

    console.print(f"\n[dim]Model  :[/dim] [green]{MODEL_NAME}[/green]")
    console.print(f"[dim]Router :[/dim] [green]llama-3.1-8b-instant[/green]")
    console.print("[dim]Commands: 'quit' • 'switch' • 'new'[/dim]\n")
    console.print(Rule(style="dim"))

    # each session gets a unique id based on time so memory is separate
    session_id = f"session_{int(time.time())}"
    config = {"configurable": {"thread_id": session_id}}

    while True:
        user_input = console.input("\n[bold green]you:[/bold green] ").strip()

        # exit the program
        if user_input.lower() == "quit":
            console.print(Panel.fit("[dim]Goodbye![/dim]", border_style="dim"))
            break

        # switch to a different model — rebuilds the whole agent
        if user_input.lower() == "switch":
            model, MODEL_NAME = select_model()
            agent = build_graph(model, router_model, memory)
            session_id = f"session_{int(time.time())}"
            config = {"configurable": {"thread_id": session_id}}
            console.print(f"[dim]Switched to[/dim] [green]{MODEL_NAME}[/green]")
            console.print(Rule(style="dim"))
            continue

        # start a fresh conversation — old memory is cleared
        if user_input.lower() == "new":
            session_id = f"session_{int(time.time())}"
            config = {"configurable": {"thread_id": session_id}}
            console.print("[dim]Started a new conversation — previous context cleared.[/dim]")
            console.print(Rule(style="dim"))
            continue

        if not user_input:
            continue

        try:
            # stream_mode="messages" gives us AIMessageChunks one token at a time
            # this means the response starts printing immediately instead of waiting
            console.print(Rule(style="cyan"))
            console.print("[bold cyan]Assistant:[/bold cyan]")

            def stream_response(input_data):
                """Stream tokens and print them as they arrive."""
                for chunk, metadata in agent.stream(
                    input_data,
                    config=config,
                    stream_mode="messages"
                ):
                    # only print AI text chunks — skip tool call chunks and routing decisions
                    if (hasattr(chunk, "content")
                            and isinstance(chunk.content, str)
                            and chunk.content
                            and not getattr(chunk, "tool_call_chunks", None)
                            and metadata.get("langgraph_node") != "summarize"):
                        print(chunk.content, end="", flush=True)
                print()  # newline once the stream ends

            stream_response({"messages": [HumanMessage(content=user_input)]})
            console.print(Rule(style="cyan"))

            # check if the graph paused mid-run waiting for human confirmation
            # this happens when execute_tools_node hits a DANGEROUS_TOOLS call
            graph_state = agent.get_state(config)
            while graph_state.tasks and any(t.interrupts for t in graph_state.tasks):
                # pull the message that interrupt() sent us
                interrupt_msg = graph_state.tasks[0].interrupts[0].value
                console.print(f"\n[bold yellow]⚠  Confirmation required:[/bold yellow]")
                console.print(f"[yellow]{interrupt_msg}[/yellow]")
                answer = console.input("[bold yellow]your answer:[/bold yellow] ").strip()

                # Command(resume=answer) sends the answer back into the graph
                # the graph picks up exactly where interrupt() was called
                console.print(Rule(style="cyan"))
                console.print("[bold cyan]Assistant:[/bold cyan]")
                stream_response(Command(resume=answer))
                console.print(Rule(style="cyan"))

                # check again — there might be more dangerous tools in the same request
                graph_state = agent.get_state(config)

        except KeyboardInterrupt:
            # user pressed Ctrl+C to stop a slow response
            console.print("\n[dim]Stopped. Type 'new' to clear context or keep chatting.[/dim]")

        except Exception as e:
            console.print(Panel(
                f"[red]Error:[/red] Something went wrong.\n"
                f"Try rephrasing or type 'switch' to change models.\n\n[dim]{str(e)[:150]}[/dim]",
                title="[red]Error[/red]",
                border_style="red"
            ))
