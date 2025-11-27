#!/bin/bash

# Get the directory where this script is located and change to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT" || {
    echo "Error: Could not change to project root: $PROJECT_ROOT"
    exit 1
}

echo "Select which docker-compose configuration to use:"
echo ""
echo "1) Mac (docker-compose-mac.yaml)"
echo "2) Linux/NVIDIA (docker-compose.yaml)"
echo ""
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo "Starting with Mac configuration..."
        docker-compose -f docker-compose-mac.yaml up
        ;;
    2)
        echo "Starting with Linux/NVIDIA configuration..."
        docker-compose -f docker-compose.yaml up
        ;;
    *)
        echo "Invalid choice. Please run the script again and select 1 or 2."
        exit 1
        ;;
esac

