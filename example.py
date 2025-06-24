import asyncio
from worker import Sprocket, Runner


class ExampleSprocket(Sprocket):
    def setup(self) -> None:
        self.multiplier = 0.3

    def predict(self, args: dict) -> dict:
        value = args["multiplicand"] * self.multiplier
        return {"value": value}


if __name__ == "__main__":
    asyncio.run(Runner(ExampleSprocket(), "example-org/example-model-name").run())
