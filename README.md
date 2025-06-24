# Sprocket

Build inference workers for Together's Dedicated Containers. You implement `setup()` and `predict()` — Sprocket handles the queue, file transfers, and HTTP server.

## Installation

```shell
pip install sprocket --extra-index-url https://pypi.together.ai/
```

## Example

```python
import sprocket

class MyModel(sprocket.Sprocket):
    def setup(self):
        self.model = load_your_model()

    def predict(self, args: dict) -> dict:
        return {"output": self.model(args["input"])}

if __name__ == "__main__":
    sprocket.run(MyModel(), "my-org/my-model")
```

Deploy with the Jig CLI:

```shell
together beta jig deploy
```

Together provisions GPUs, handles autoscaling, and routes jobs to your workers.
