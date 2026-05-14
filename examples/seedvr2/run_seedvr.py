from pathlib import Path
import shutil

import sprocket
from huggingface_hub import snapshot_download
from projects import inference_seedvr2_3b as inf

class SeedVR2(sprocket.Sprocket):
    def setup(self) -> None:
        print("Downloading SeedVR2-3B checkpoints from Hugging Face …", flush=True)
        snapshot_download(
            repo_id="ByteDance-Seed/SeedVR2-3B",
            local_dir="ckpts",
            allow_patterns=["seedvr2_ema_3b.pth", "ema_vae.pth"],
        )

        self.runner = inf.configure_runner(sp_size=1)

    def predict(self, args: dict) -> dict:
        media = args.get("media") or args.get("video_path")
        if media is None:
            raise ValueError("`media` required.")

        infile = Path(media)
        inp = infile.parent.resolve()
        out = Path("output").resolve()
        outfile = out / infile.name
        shutil.rmtree(out, ignore_errors=True)
        kv = dict(
            video_path=str(inp),
            output_dir=str(out),
            batch_size=1,
            cfg_scale=float(args.get("cfg_scale", 1.0)),
            cfg_rescale=float(args.get("cfg_rescale", 0.0)),
            sample_steps=int(args.get("sample_steps", 1)),
            seed=int(args.get("seed", 666)),
            res_h=int(args.get("res_h", 720)),
            res_w=int(args.get("res_w", 1280)),
            sp_size=1,
            out_fps=None if (v := args.get("out_fps")) is None else float(v),
        )
        inf.generation_loop(self.runner, **kv)

        return {"output": sprocket.FileOutput(outfile)}

if __name__ == "__main__":
    sprocket.run(SeedVR2())
