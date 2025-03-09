#!/bin/bash

# AI Grid Deployment Script for Render
# This script helps prepare your project for deployment to Render

echo "=== AI Grid Deployment Preparation ==="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install Git first."
    exit 1
fi

# Ensure we're in a git repository
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
fi

# Commit any changes
if [ -n "$(git status --porcelain)" ]; then
    echo "Committing changes..."
    git add .
    git commit -m "Prepare for Render deployment"
fi

echo ""
echo "=== Deployment Preparation Complete ==="
echo ""
echo "To deploy to Render:"
echo ""
echo "1. Go to https://dashboard.render.com/select-repo"
echo "2. Connect your GitHub/GitLab repository"
echo "3. Select 'Blueprint' as the deployment type"
echo "4. Render will detect the render-blueprint.yaml file"
echo "5. If you already have a service named 'ai-grid-frontend', delete it first"
echo "6. Apply the blueprint to create the services with the correct names"
echo ""
echo "Your application will be available at:"
echo "- Frontend: https://ai-grid.onrender.com"
echo "- Backend API: https://ai-grid-backend.onrender.com"
echo ""
echo "Note: It may take a few minutes for the services to build and deploy."
