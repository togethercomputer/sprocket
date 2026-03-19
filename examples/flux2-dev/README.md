# FLUX.2-dev Example

Serves [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) as an image generation endpoint using Sprocket.

## How to Deploy

1. Generate a unique deployment name and update `pyproject.toml`:

   ```bash
   sed -i '' "s/^name = \"sprocket-flux2-dev\"/name = \"flux2-dev-$(date +%s)\"/" pyproject.toml
   ```

2. Set your Hugging Face token as a secret (required to download the gated model weights):

   ```bash
   together beta jig secrets set --name HF_TOKEN --value <your-hf-token>
   ```

3. Deploy:

   ```bash
   together beta jig deploy
   ```

4. Submit a generation request:

   ```bash
   together beta jig submit --prompt "a cat sitting on a red chair"
   ```

   You can follow the logs while the job is processing:

   ```bash
   together beta jig logs --follow
   ```

5. Check the job output:

   ```bash
   together beta jig job-status --request-id <request-id>
   ```

   The response contains a base64-encoded PNG image under `outputs.image`.
