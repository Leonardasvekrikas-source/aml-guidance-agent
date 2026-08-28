# Python 3.12, not 3.13+: torch and sentence-transformers wheels lag the
# newest CPython release, and a portfolio repo that will not pip install is
# worthless. Pinned deliberately.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# poppler-utils is a fallback text extractor for PDFs whose text layer pypdf
# handles badly; curl is used by the corpus download script.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits do not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "aml_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
