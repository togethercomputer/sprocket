FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv
ADD https://github.com/krallin/tini/releases/download/v0.19.0/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]
RUN uv pip install --system httpx torch transformers torchvision pillow starlette uvicorn
RUN uv pip install --system orjson
RUN --mount=type=cache,target=/root/.cache/huggingface uv run --with "huggingface_hub" huggingface-cli download kaiko-ai/midnight --local-dir /app/kaiko-ai/midnight
WORKDIR /app
COPY worker.py examples/midnight/midnight.py /app/
ENV MODEL_PATH=kaiko-ai/midnight
CMD ["/usr/local/bin/python3", "midnight.py"]
