FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv
RUN uv pip install --system "aiohttp[speedups]"
WORKDIR /app
COPY worker.py example.py /app/
ENTRYPOINT ["/usr/local/bin/python3", "example.py"]
