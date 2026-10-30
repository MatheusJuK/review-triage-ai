FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir -e .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn review_triage.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
