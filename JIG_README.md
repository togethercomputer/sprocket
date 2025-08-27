# jig

Beautiful deployment tool for Together AI. Like `flyctl` but simpler.

## Philosophy

One file. Zero dependencies. Just works.

## Usage

```bash
export TOGETHER_API_KEY=your_key_here
jig deploy
```

That's it. Everything else is automatic.

## Configuration

Add to your `pyproject.toml`:

```toml
[tool.jig]
name = "my-model"  # optional, defaults to directory name

[tool.jig.image]
python_version = "3.11"
system_packages = ["git", "libglib2.0-0"]
cmd = "python app.py"
auto_include_git = true  # copy all git-tracked files

[tool.jig.deploy]
name = "my-model"
gpu_type = "h100-80gb"
port = 8000
```

## Commands

- `jig init` - Create initial config
- `jig dockerfile` - Generate Dockerfile
- `jig build` - Build image
- `jig push` - Push to registry
- `jig set_secret --name secret_name --value secret_content --env_var env_var_name` - Create secret and map it to environment variable
- `jig unset_secret --name secret_name` - Remove secret mapped to the deployment
- `jig list_secrets` - List all secrets mapped to deployment
- `jig deploy` - Build, push, and deploy
- `jig deploy --image existing:tag` - Deploy existing image
- `jig deploy --build-only` - Build and push only
- `jig status` - Show deployment status
- `jig destroy` - Delete deployment

## How it works

1. Reads config from `pyproject.toml`
2. Generates optimized multi-stage Dockerfile with uv
3. Builds with Docker BuildKit caching
4. Pushes to Together registry
5. Creates/updates deployment via API

Files are copied based on git tracking (repo must be clean) plus any explicit `copy` entries.

The Dockerfile is regenerated when `pyproject.toml` changes. Otherwise builds are unconditional like `fly deploy`.

## State

Minimal state in `.jig.json`:
- `deployment_id` - Current deployment
- `username` - Registry username from API

No complex build tracking. Filesystem timestamps are truth.
