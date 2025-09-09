FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv
COPY --from=krallin/ubuntu-tini:latest /usr/local/bin/tini /tini
ENTRYPOINT ["/tini", "--"]
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=bind,src=./dist,dst=/root/dist uv pip install --system /root/dist/sprocket-0.1-py3-none-any.whl torch transformers torchvision pillow
RUN --mount=type=cache,target=/root/.cache/huggingface uv run --with "huggingface_hub" huggingface-cli download kaiko-ai/midnight --local-dir /app/kaiko-ai/midnight
WORKDIR /app
ENV MODEL_PATH=kaiko-ai/midnight
ADD midnight.py /app
CMD ["/usr/local/bin/python3", "midnight.py"]
