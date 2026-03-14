# docker/model-servers/Dockerfile
#
# Single Dockerfile for all four model servers.
#
# Build args
# ----------
# SERVER            (required) one of: embedder | similarity | als | metadata
# REQUIREMENTS_FILE (required) one of: embedder.txt | similarity.txt | als.txt | metadata.txt
#
# Both are set in docker-compose.yml under build.args and environment.
#
# Build context must be the project root so model_servers/, models/, and
# requirements/ are all available. docker-compose.yml sets context: ../..
#
# Artifacts (models/artifacts/) are NOT baked into the image. They are
# mounted read-only at /app/models/artifacts at runtime so a retrained
# model can be swapped without rebuilding the image.

ARG PYTHON_VERSION=3.11


# ---------------------------------------------------------------------------
# Stage 1: install Python dependencies into an isolated venv
# ---------------------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim AS builder

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Build tools needed by some packages (e.g. numpy C extensions).
# Kept in the builder stage only — not copied to the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	gcc \
	&& rm -rf /var/lib/apt/lists/*

ARG REQUIREMENTS_FILE
COPY requirements/base.txt /tmp/base.txt
COPY requirements/${REQUIREMENTS_FILE} /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# ---------------------------------------------------------------------------
# Stage 2: lean runtime image
# ---------------------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim AS runtime

# libgomp1: OpenMP runtime required by faiss-cpu and numpy at inference time.
# curl: used by the Docker HEALTHCHECK only.
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgomp1 \
	curl \
	&& rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"


WORKDIR /app

COPY model_servers/ ./model_servers/
COPY models/ ./models/
COPY gunicorn_model_server.conf.py ./gunicorn.conf.py


ARG SERVER
ENV SERVER=${SERVER}
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
	CMD curl -fs http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "exec gunicorn model_servers.${SERVER}.main:app --config /app/gunicorn.conf.py"]
