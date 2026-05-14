"""Minimal sprocket worker"""

import sprocket


class HelloWorld(sprocket.Sprocket):
    def setup(self) -> None:
        self.greeting = "Hello"

    def predict(self, args: dict) -> dict:
        name = args.get("name", "world")
        return {"message": f"{self.greeting}, {name}!"}


if __name__ == "__main__":
    sprocket.run(HelloWorld())
