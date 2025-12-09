FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS builder

RUN --mount=type=cache,target=/var/cache/apt-builder \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive \
  apt-get install -y --no-install-recommends git libglib2.0-0 python3.12 && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml .

# Create virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --compile-bytecode --group production .

RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,src=./dist,dst=/root/dist \
  uv pip install /root/dist/sprocket-0.1-py3-none-any.whl

FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 

RUN --mount=type=cache,target=/var/cache/apt-final \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive \
  apt-get install -y --no-install-recommends build-essential git libglib2.0-0 python3.12 libnuma1 && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=krallin/ubuntu-tini:latest /usr/local/bin/tini /tini
ENTRYPOINT ["/tini", "--"]

WORKDIR /app
COPY app.py app.py
RUN --mount=type=bind,source=.,target=/src cp /src/.worker.p* worker.py 2>/dev/null || true

CMD ["python", "app.py"]
