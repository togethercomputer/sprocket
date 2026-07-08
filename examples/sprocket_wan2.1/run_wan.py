import sprocket


def make_sprocket() -> sprocket.Sprocket:
    from wan_sprocket import WanSprocket

    return WanSprocket()


if __name__ == "__main__":
    # pass a factory instead of Sprocket instance to the runner so that parent process doesn't import libraries it doesn't need
    sprocket.run(make_sprocket, use_torchrun=True)
