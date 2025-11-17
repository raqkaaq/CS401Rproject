#!/bin/bash

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

