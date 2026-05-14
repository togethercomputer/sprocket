# Speech-to-Speech Example

A real-time speech-to-speech application that processes audio input and responds with synthesized speech.

## How to Run

1. Generate a unique deployment name and update `pyproject.toml`:

   ```bash
   sed -i '' "s/^name = \"speech-to-speech\"/name = \"speech-to-speech-$(date +%s)\"/" pyproject.toml
   ```

2. Install client dependencies:

   ```bash
   uv sync --no-default-groups --group client
   ```

3. Deploy the application:

   ```bash
   together beta jig deploy
   ```

4. Run the client with your deployment ID:

   ```bash
   DEPLOYMENT_ID=<deployment_id> python sts_client.py
   ```

5. **Usage:**
   - Speak your query
   - Press Enter to send the audio
   - The response will be automatically spoken back to you

## Requirements

The client dependencies are managed in the "client" group and can be installed using `uv sync --group client`.
