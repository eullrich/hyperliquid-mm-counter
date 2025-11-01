FROM python:3.11-slim

# Install PostgreSQL client for healthchecks
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY static ./static
COPY migrations ./migrations

# Create logs directory
RUN mkdir -p /app/logs

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
