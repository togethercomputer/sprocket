# Kitchen Sink Example

A no-GPU Sprocket worker that exercises every platform feature: volume mounts, secrets, environment variables, file I/O, `FileOutput` uploads, and queued job processing.

See [`kitchen_sink.py`](./kitchen_sink.py) for the worker logic.

## How to Deploy

1. Generate a unique deployment name and update `pyproject.toml`:

   ```bash
   sed -i '' "s/^name = \"kitchen-sink\"/name = \"kitchen-sink-$(date +%s)\"/" pyproject.toml
   ```

2. Create the `kitchen-sink-pantry` volume and upload recipe files (the worker reads `*.txt` from `/pantry` at startup):

   ```bash
   together beta jig volumes create --name kitchen-sink-pantry --source ./pantry
   ```

3. Set the `SECRET_SPICE` secret (gets injected as an env var):

   ```bash
   together beta jig secrets set --name SECRET_SPICE --value cumin
   ```

4. Deploy:

   ```bash
   together beta jig deploy
   ```

5. Submit a request that exercises every code path:

   ```bash
   together beta jig submit --payload '{"dish":"pasta","menu":true,"read_env":"SECRET_SPICE","read_file":"/pantry/curry.txt","receipt":true,"sleep":1}' --watch
   ```

   Drop `--watch` to grab the request id and tail logs in parallel:

   ```bash
   together beta jig logs --follow
   together beta jig job-status --request-id <request-id>
   ```

   The response includes:
   - `plate` — string combining the requested `dish`, its recipe from the volume, and the secret spice
   - `menu` — sorted list of recipe names found in `/pantry`
   - `env_value` — value of the env var named by `read_env`
   - `file_content` — text content at `read_file`
   - `receipt` — uploaded `FileOutput` URL containing a generated receipt
