import base64
import logging
import os
from io import BytesIO

import sprocket
import torch
from diffusers import Flux2Pipeline

logging.basicConfig(level=logging.INFO)


class Flux2Sprocket(sprocket.Sprocket):
    def setup(self) -> None:
        model = "diffusers/FLUX.2-dev-bnb-4bit"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Loading Flux2 pipeline from {model} on {device}...")
        pipe = Flux2Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
        self.pipe = pipe.to(device)
        logging.info("Pipeline loaded successfully!")

    def predict(self, args: dict) -> dict:
        prompt = args.get("prompt", "a cat")

        # Optional parameters with defaults
        num_inference_steps = args.get("num_inference_steps", 28)
        guidance_scale = args.get("guidance_scale", 4.0)

        # Generate image
        logging.info(f"Generating image for prompt: {prompt[:50]}...")
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        logging.info("Image generated successfully")

        return {"image": img_str, "format": "png", "encoding": "base64"}


if __name__ == "__main__":
    queue_name = os.environ.get("TOGETHER_DEPLOYMENT_NAME", "sprocket-flux2-dev")
    sprocket.run(Flux2Sprocket(), queue_name)
