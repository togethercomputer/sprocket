# Sprocket

Sprocket is a lightweight Python SDK to help you deploy and serve custom models on the Together Drivetrain platform, handling all the backend communication and job management for you. You just need to implement your model logic and run the provided job loop.

## Purpose:

Sprocket is designed to help you deploy and serve models by connecting to Together's infrastructure, handling job retrieval, execution, and status updates.

## How it works:

You define your own model logic by subclassing the `Sprocket` class and implementing the `setup` and `predict` methods.
The Runner class manages communication with Together's job queue, fetching jobs, running your model's `predict` method, and reporting results back.

## Example Usage:

The `example.py` file shows how to create a custom model by subclassing `Sprocket`, then running it with `Runner`. The example multiplies an input value by a constant.
