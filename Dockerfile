# API image. Only app dependencies — the RAGAS/LangChain eval stack is deliberately
# NOT installed here (DECISIONS.md D5 keeps it in a separate venv), which keeps this
# image small and stops an eval-only dependency conflict from breaking the service.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, as their own layer: application edits then rebuild in seconds
# instead of re-resolving the dependency tree every time.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

# Non-root: the container never needs to write to its own filesystem.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# No --reload: that's a development flag and would watch files this image doesn't mount.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
