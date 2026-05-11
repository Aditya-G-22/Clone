#!/bin/bash

# Install dependencies
uv sync

# Copy env template
cp .env.example .env

echo "Setup complete!"
echo "Add your Anthropic API key to .env and run: uv run python main.py"
