# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install pip requirements as non-root user
COPY requirements.txt .
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org && \
    chown -R appuser:appgroup /app && \
    chmod -R g+w /app

# Copy application code
COPY . .

# Switch to non-root user
USER appuser

# Expose port for application
EXPOSE 8000

# Start the application server using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]







