API_URL?=https://api.together.ai
REGISTRY_URL?=registry.together.xyz

check:
	@test -n "$(TOGETHER_API_KEY)" || (echo "ERROR: TOGETHER_API_KEY not set" && exit 1)
	@test -n "$(USERNAME)" || (echo "ERROR: USERNAME not set" && exit 1)

.docker_built: Dockerfile
	docker build --platform linux/amd64 -t "$(REGISTRY_URL)/$(USERNAME)/wan2.1" .
	touch .docker_built

run: .docker_built
	docker run --gpus all -e TOGETHER_API_KEY=$(TOGETHER_API_KEY) -v $(shell pwd):/workspace/ -v /scratch/hf_cache:/root/.cache/huggingface --net host --ulimit memlock=-1 --ulimit stack=67108864 -it "$(REGISTRY_URL)/$(USERNAME)/wan2.1"

.push: check .docker_built
	echo $(TOGETHER_API_KEY) | docker login $(REGISTRY_URL) --username user --password-stdin
	docker push "$(REGISTRY_URL)/$(USERNAME)/wan2.1:latest"
	touch .push

# create a deployment and write the deployment id to a file using the /v1/deployments endpoint
# define block to work around newline issues
define DEPLOYMENT_JSON
{
  "name": "Wan 2.1",
  "description": "Wan2.1 SQS example",
  "image": "$(REGISTRY_URL)/$(USERNAME)/wan2.1",
  "environment_variables": {
    "TOGETHER_API_BASE_URL": "$(API_URL)",
    "TOGETHER_API_KEY": "$(TOGETHER_API_KEY)"
  },
  "min_replicas": 1,
  "max_replicas": 1,
  "port": 8000,
  "gpu_type": "h100-80gb",
  "gpu_count": 2,
  "cpu": 1
}
endef

deployment_id.txt: .push
	curl "$(API_URL)/v1/deployments" \
	     -H 'Content-Type: application/json' \
	     -H "Authorization: Bearer $(TOGETHER_API_KEY)" \
	     -d '$(shell echo '$(DEPLOYMENT_JSON)' | tr -d '\n')'  \
	     | tee /tmp/deployment_output.json | jq -r '.id' > deployment_id.txt
	@cat /tmp/deployment_output.json | jq

# get deployment status including desired_replcias and ready_replicas
deployment_status: deployment_id.txt
	curl "$(API_URL)/v1/deployments/$(shell cat deployment_id.txt)" \
	    -H 'Content-Type: application/json' \
	    -H "Authorization: Bearer $(TOGETHER_API_KEY)" | jq

# get queue length and running jobs
queue_status: check
	curl -H 'Content-Type: application/json' -H "Authorization: Bearer $(TOGETHER_API_KEY)" \
	    "$(API_URL)/internal/v1/queue/status?model=example-org/example-model-name" | jq

# submit a job
submit: check
	curl "$(API_URL)/v1/videos/generations" \
	-H 'Content-Type: application/json' \
	-H "Authorization: Bearer $(TOGETHER_API_KEY)" \
	-d '{ "model": "example-org/example-model-name", "payload": { "prompt": "A cat jumping into the stars" }, "priority": 1 }'

# submit ten jobs, then watch deployment_status and queue_status
monitor: deployment_id.txt
	for i in {1..3}; do \
		echo "Submitting job $$i"; \
		make submit; \
		sleep 1; \
	done; \
	@echo "Monitoring deployment and queue status. Press Ctrl+C to stop."; \
	while true; do \
		echo "Deployment status:"; \
		make deployment_status; \
		echo "Queue status:"; \
		make queue_status; \
		sleep 5; \
	done
destroy:
	@cat deployment_id.txt || echo "deployment not created, nothing to do" && exit 0
	curl -X DELETE "$(API_URL)/v1/deployments/$(shell cat deployment_id.txt)" \
	    -H 'Content-Type: application/json' \
	    -H "Authorization: Bearer $(TOGETHER_API_KEY)"
	mv deployment_id.txt deleted_deployment_id.txt

build: .docker_built
push: .push
deploy: deployment_id.txt
