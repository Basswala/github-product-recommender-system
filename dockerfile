## Parent image
FROM python:3.10-slim

## Essential environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

## Work directory inside the docker container
WORKDIR /app

## Installing system dependancies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

## Install uv
RUN pip install uv

## Copying ur all contents from local to app
COPY . .

## Install dependencies with uv
RUN uv sync

# Used PORTS
EXPOSE 3000

# Run the app 
CMD ["python", "app.py"]