# SQS Example

If you currently use SQS and don't want to fully integrate Sprocket, you can keep using boto3, as shown in this example. Check out `Makefile` for the commands to deploy.

You can run `make deploy` to automatically build and deploy Wan2.1 with SQS on Together. `TOGETHER_API_KEY` and `USERNAME` must be set. `make monitor` will build and deploy if needed, submit 10 jobs, and then watch the deployment and queue state.
