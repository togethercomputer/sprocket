# SeedVR2-3B (Sprocket)

Runs [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) **`projects.inference_seedvr2_3b`** via **`configure_runner`** / **`generation_loop`** in **`run_seedvr.py`**.

## Deploy

1. Unique deployment id (writes **`pyproject.toml`**):

   ```bash
   sed -i '' "s/^name = \"sprocket-seedvr2-3b\"/name = \"seedvr2-$(date +%s)\"/" pyproject.toml
   ```

2. Deploy:

   ```bash
   together beta jig deploy
   ```

## Example request

```bash
together beta jig submit --payload '{
  "media": "https://huggingface.co/datasets/Iceclear/SeedVR_VideoDemos/resolve/main/seedvr2_videos/1_21_lq.mp4",
  "sample_steps": 1,
  "cfg_scale": 1.0,
  "seed": 666
}' --watch
```
