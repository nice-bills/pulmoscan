FROM python:3.12-slim

# Install system dependencies
# libgl1/libglib2.0 are often needed for image processing libraries (opencv/pillow extensions)
# gcc/libpq-dev are needed for building some python extensions (like psycopg2 if no binary wheel)
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies
# --system installs into the system python, avoiding venv complexity in Docker
# --deploy ensures consistent versions from uv.lock
RUN uv pip install --system --no-deps -r pyproject.toml

# Copy the rest of the application
COPY . .

# Default command (overridden in docker-compose for the worker)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
