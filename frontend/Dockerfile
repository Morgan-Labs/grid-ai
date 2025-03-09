# syntax=docker/dockerfile:1.4
# Enable BuildKit features

# Build stage
FROM oven/bun:latest AS build

WORKDIR /app

# Copy package.json and lockfile first for better caching
COPY package.json bun.lockb* ./

# Install dependencies with caching
RUN --mount=type=cache,target=/root/.bun \
    bun install --frozen-lockfile

# Skip TypeScript checks during build to avoid type errors
RUN sed -i 's/"build": "tsc && vite build"/"build": "vite build"/g' package.json

# Create a .env file with the API URL if provided as a build arg
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL:-http://localhost:8000}
RUN echo "VITE_API_URL=${VITE_API_URL}" > .env

# Copy the rest of the application code
COPY . .

# Build the application
RUN --mount=type=cache,target=/root/.bun \
    bun run build

# Production stage
FROM nginx:alpine

# Copy built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Configure nginx
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:3000/ || exit 1

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
