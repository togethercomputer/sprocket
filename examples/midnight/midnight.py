import base64
import io
import logging
import math
import mmap
import os
import time

import httpx
import numpy as np
import sprocket
import torch
from PIL import Image
from torchvision.io import image
from torchvision.transforms import v2
from transformers import AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MidnightSprocket(sprocket.Sprocket):
    def setup(self) -> None:
        model_path = os.getenv("MODEL_PATH")
        if not model_path:
            raise RuntimeError("MODEL_PATH environment variable not set")

        if not os.path.exists(model_path):
            raise RuntimeError(f"Model path does not exist: {model_path}")

        logger.info(f"Loading model from {model_path}")
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()

        self.transform = v2.Compose(
            [
                v2.Resize(224),
                v2.CenterCrop(224),
                v2.ToTensor(),
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.http_client = httpx.Client(timeout=30.0)

        logger.info(f"Model loaded successfully on {self.device}")

    def predict(self, args: dict) -> dict:
        task = args.get("task")
        images = args["images"]

        if task not in ["classify", "segment"]:
            raise ValueError("task must be 'classify' or 'segment'")

        batch_tensor = self.load_and_transform_images(images)

        with torch.inference_mode():
            output = self.model(batch_tensor)

        if task == "classify":
            embeddings = self.extract_classification_embedding(output.last_hidden_state)
        else:
            embeddings = self.extract_segmentation_embedding(output.last_hidden_state)
        fmt = args.get("format", "json")
        if fmt == "json":
            result = {"embeddings": embeddings.cpu().numpy().tolist()}
        elif fmt == "npy,base64":
            buf = io.BytesIO()
            np.save(buf, embeddings.cpu().numpy())
            buf.seek(0)
            result = {"embeddings_npy_base64": base64.b64encode(buf.read())}
        else:
            raise ValueError('format must be "json" or "npy,base64"')
        return result

    def load_image_from_source(self, image_source: str) -> Image.Image:
        if not isinstance(image_source, str) or not image_source.strip():
            raise ValueError("images must be a non-empty list")
        if image_source.startswith("data:image"):
            data = image_source.split(",", 1)[1]
            image_data = base64.b64decode(data)
        elif image_source.startswith(("http://", "https://")):
            response = self.http_client.get(image_source)
            response.raise_for_status()
            image_data = response.content
        else:
            raise ValueError(
                "Image source must be either a data URL (base64) or HTTP/HTTPS URL"
            )

        f = open("/tmp/img", "wb+")
        f.truncate(len(image_data))
        mmap.mmap(f.fileno(), len(image_data)).write(image_data)
        return image.read_image("/tmp/img", mode=image.ImageReadMode.RGB)

    def load_and_transform_images(self, image_sources: list[str]) -> torch.Tensor:
        st = time.time()
        images = [
            self.load_image_from_source(image_source) for image_source in image_sources
        ]
        st_2 = time.time()
        tensors = self.transform(images)
        logging.info(
            f"Took {st_2-st:.3f}s to download, {time.time()-st_2:.3f}s to transform"
        )
        return torch.stack(tensors).to(self.device)

    def extract_classification_embedding(self, tensor):
        cls_embedding, patch_embeddings = tensor[:, 0, :], tensor[:, 1:, :]
        return torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)

    def extract_segmentation_embedding(self, tensor):
        features = tensor[:, 1:, :].permute(0, 2, 1)
        batch_size, hidden_size, patch_grid = features.shape
        height = width = int(math.sqrt(patch_grid))
        return features.view(batch_size, hidden_size, height, width)


if __name__ == "__main__":
    queue_name = os.environ.get("TOGETHER_DEPLOYMENT_NAME", "midnight")
    sprocket.run(MidnightSprocket(), queue_name)
