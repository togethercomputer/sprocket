# Qwen2.5-VL vLLM Example

Serves [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)
via a pre-built [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai) Docker
image with an OpenAI-compatible API.

This example demonstrates HTTP server mode without Sprocket. Instead of
implementing a `Sprocket` subclass, the `pyproject.toml` specifies a pre-built
image and a `command` that starts vLLM directly. Jig deploys it as-is — no
`docker build` step. Requests go directly to the HTTP endpoint rather than
through the job queue, so use `jig endpoint` + `curl` (or any OpenAI client)
instead of `jig submit`.

## How to Run

1. Deploy:

   ```bash
   together beta jig deploy
   ```

2. Get the endpoint URL:

   ```bash
   together beta jig endpoint
   ```

3. Send a request:

   ```bash
   curl $(together beta jig endpoint)/v1/chat/completions \
     -H "Authorization: Bearer $TOGETHER_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen2.5-VL-32B-Instruct",
       "messages": [
         {
           "role": "user",
           "content": [
             {"type": "video_url", "video_url": {"url": "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"}},
             {"type": "text", "text": "Describe this video."}
           ]
         }
       ]
     }'
   ```

   Response:

   ```json
   {
     "choices": [
       {
         "index": 0,
         "message": {
           "role": "assistant",
           "content": "This video showcases the features and benefits of a transparent plastic fruit packaging container, ..."
         }
       }
     ]
   }
   ```
