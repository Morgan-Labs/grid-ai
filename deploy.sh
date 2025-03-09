#!/bin/bash

# Script to deploy changes to Render via GitHub

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting deployment process...${NC}"

# Check if there are any changes to commit
if [[ -z $(git status -s) ]]; then
  echo -e "${RED}No changes to commit. Exiting.${NC}"
  exit 1
fi

# Show the changes that will be committed
echo -e "${YELLOW}Changes to be committed:${NC}"
git status -s

# Ask for confirmation
read -p "Do you want to commit and push these changes? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo -e "${RED}Deployment cancelled.${NC}"
  exit 1
fi

# Ask for commit message
echo -e "${YELLOW}Enter commit message:${NC}"
read -r commit_message

# Commit the changes
echo -e "${YELLOW}Committing changes...${NC}"
git add .
git commit -m "$commit_message"

# Push to GitHub
echo -e "${YELLOW}Pushing to GitHub...${NC}"
git push

# Check if push was successful
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Changes pushed successfully!${NC}"
  echo -e "${YELLOW}Render will automatically deploy the changes.${NC}"
  echo -e "${YELLOW}You can check the deployment status at: https://dashboard.render.com/${NC}"
else
  echo -e "${RED}Failed to push changes to GitHub.${NC}"
  exit 1
fi

echo -e "${GREEN}Deployment process completed!${NC}"
