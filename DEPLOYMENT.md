# Deploying AI Grid to Render with Milvus Lite

This guide provides step-by-step instructions for deploying the AI Grid application to Render using Milvus Lite as the vector database.

## Prerequisites

1. A [Render](https://render.com) account
2. An [OpenAI API key](https://platform.openai.com/api-keys)
3. Git repository with your AI Grid code

## Deployment Steps

### 1. Run the Deployment Preparation Script

Run the deployment preparation script to ensure your code is ready for deployment:

```bash
./deploy-to-render.sh
```

This script will:
- Initialize a Git repository if needed
- Commit any changes
- Provide instructions for deploying to Render

### 2. Push Your Code to a Git Repository

Push your code to a Git repository (GitHub, GitLab, etc.):

```bash
git remote add origin <your-repository-url>
git push -u origin main
```

### 3. Deploy Using Render Blueprint

1. Go to [Render Dashboard](https://dashboard.render.com/select-repo)
2. Connect your GitHub/GitLab repository
3. Select "Blueprint" as the deployment type
4. Render will detect the `render-blueprint.yaml` file
5. **Important**: If you already have a service named 'ai-grid-frontend', delete it first
6. Apply the blueprint to create the services with the correct names

### 4. Configure Environment Variables

For the backend service, you'll need to set these environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key is already set in the backend/.env file
- `MILVUS_DB_URI`: This is already set to `/data/milvus_db.db` in the render-blueprint.yaml file

## Accessing Your Deployed Application

Once deployment is complete, you can access your application at:

- Frontend: `https://ai-grid.onrender.com`
- Backend API: `https://ai-grid-backend.onrender.com`
- API Documentation: `https://ai-grid-backend.onrender.com/docs`

## UI Improvements

The deployment includes modifications to reduce unwanted pop-ups:
- Simplified UI components
- Removed error test modal
- Streamlined controls

## Troubleshooting

- Check service logs in the Render dashboard
- Verify environment variables are set correctly
- Ensure the persistent disk is properly mounted and accessible
- Check that the Milvus Lite database file is being created in the `/data` directory

## Scaling

To scale your application:

1. Go to the service in your Render dashboard
2. Click on "Settings"
3. Adjust the plan and instance count as needed
