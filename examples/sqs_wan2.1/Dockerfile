FROM python:3.11-slim

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends git libglib2.0-0 libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --link --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system \
      diffusers===0.33.0 \
      transformers>=4.44.0 \
      para_attn \
      ftfy accelerate einops omegaconf pillow ffmpeg-python opencv-python \
      boto3

WORKDIR /workspace
COPY --link server.py run_wan.py .
# used for the healthcheck and metrics
RUN mkdir http && touch http/health && echo "utilization 0.0" > http/metrics
CMD ["bash", "-c", "torchrun --standalone --nproc_per_node=2 run_wan.py"]
