import json
import os
import pathlib
from typing import Optional

import torch
import torch.distributed as dist
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

import sprocket


class WanSprocket(sprocket.Sprocket):
    def setup(self) -> None:
        dist.init_process_group()
        torch.cuda.set_device(dist.get_rank())

        pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.pipe = pipe.to("cuda")

        para_mesh = init_context_parallel_mesh(self.pipe.device.type)
        parallelize_pipe(self.pipe, mesh=para_mesh)

    def predict(self, args: dict) -> Optional[str]:
        video = self.pipe(
            prompt=args["prompt"],
            negative_prompt="",
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=30,
            output_type="pil" if dist.get_rank() == 0 else "pt",
        ).frames[0]

        if dist.get_rank() == 0:
            print(f"Saving video to output.mp4")
            export_to_video(video, "output.mp4", fps=15)
            return "output.mp4"


if __name__ == "__main__":
    sprocket.run(WanSprocket(), "example-org/example-model-name", use_torchrun=True)
