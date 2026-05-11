# Cursor Clone — CLI Coding Agent

A CLI coding agent powered by **Claude** (Anthropic) and **LangGraph**. Inspired by Cursor and Claude Code.

## What it does

You talk to it in the terminal. It can read, list, and write files on your machine autonomously — reasoning about what actions to take using an agentic loop.

```
you: read main.py and explain what it does
Claude: [reads main.py] This file is a LangGraph agent that...

you: add a docstring to the read_file function
Claude: [reads main.py] [writes main.py] Done! I added a docstring to read_file.
```

## How it works

```
User input → call_claude → tool_use? → execute_tools → call_claude → ... → END
```

Built with:
- **Anthropic SDK** — Claude as the reasoning engine
- **LangGraph** — stateful agent loop as a directed graph
- **3 tools** — `read_file`, `list_files`, `write_file`

## Setup

```bash
# Install dependencies
uv sync

# Add your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the agent
uv run python main.py
```

## Usage

```bash
uv run python main.py

you: list the files in the current directory
you: read main.py and summarize it
you: write a hello.py file that prints Hello World
you: quit
```
