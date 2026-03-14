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
#
# BLAS configuration
# ------------------
# numpy is built from source against the system OpenBLAS (libopenblas-dev)
# rather than using the bundled openblas64 wheel. openblas64 uses 64-bit
# integer indexing (USE_64BITINT=1) which adds overhead with no benefit at
# this scale (arrays are well under 2^31 elements). The thread count is
# capped via OPENBLAS_NUM_THREADS rather than a compile-time MAX_THREADS
# flag so it can be overridden per-container in docker-compose.yml.

ARG PYTHON_VERSION=3.11


# ---------------------------------------------------------------------------
# Stage 1: install Python dependencies into an isolated venv
# ---------------------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim AS builder

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Build tools and system OpenBLAS headers needed to compile numpy from source.
# Kept in the builder stage only — not copied to the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

ARG REQUIREMENTS_FILE
COPY requirements/base.txt /tmp/base.txt
COPY requirements/${REQUIREMENTS_FILE} /tmp/requirements.txt

# Install numpy from source so it links against the system OpenBLAS rather
# than pulling in the bundled openblas64 binary wheel.
RUN pip install --no-cache-dir --no-binary :numpy: numpy

RUN pip install --no-cache-dir -r /tmp/requirements.txt


# ---------------------------------------------------------------------------
# Stage 2: lean runtime image
# ---------------------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim AS runtime

# libgomp1:    OpenMP runtime required by faiss-cpu and numpy at inference time.
# libopenblas0: System OpenBLAS runtime linked by the numpy built in stage 1.
# curl:        Used by the Docker HEALTHCHECK only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Cap BLAS threads per container. With 4 containers on 6 vCPUs, 2 threads
# per container prevents oversubscription while still allowing parallelism
# on the heavier matrix operations. Override in docker-compose.yml if tuning
# individual servers.
ENV OPENBLAS_NUM_THREADS=2

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
