FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv
COPY --from=krallin/ubuntu-tini:latest /usr/local/bin/tini /tini
ENTRYPOINT ["/tini", "--"]
RUN uv pip install --system orjson httpx torch transformers torchvision pillow starlette uvicorn
RUN --mount=type=cache,target=/root/.cache/huggingface uv run --with "huggingface_hub" huggingface-cli download kaiko-ai/midnight --local-dir /app/kaiko-ai/midnight
WORKDIR /app
COPY --link worker.py midnight.py /app/
ENV MODEL_PATH=kaiko-ai/midnight
CMD ["/usr/local/bin/python3", "midnight.py"]
