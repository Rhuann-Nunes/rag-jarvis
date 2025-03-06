#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "WARNING: .env file not found. Make sure environment variables are set."
fi

# Check if the user wants to run in development mode
if [ "$1" = "dev" ] || [ -z "$1" ]; then
    echo "Starting in development mode..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        source venv/Scripts/activate
    fi
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Start the server
    echo "Starting server..."
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Usage: ./start.sh [dev]"
    echo "  dev    - Start in development mode (default)"
fi 