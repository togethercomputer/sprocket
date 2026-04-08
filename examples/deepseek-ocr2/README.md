# DeepSeek OCR 2 Example

## Deploy

First, make the project name in `pyproject.toml` unique.

```bash
sed -i '' "s/^name = \"deepseek-ocr2\"/name = \"deepseek-ocr2-$(date +%s)\"/" pyproject.toml
```

From within this directory, run `together beta jig deploy`.

## Inference Requests

Run `together beta jig endpoint` to get the endpoint from where the model is served.

Create your own request, or use the example request provided below after replacing the endpoint.

### Example Request

**Request**

```
curl --header "Authorization: Bearer $TOGETHER_API_KEY" \
  -H "Content-Type: application/json" \
  {TOGETHER DCI ENDPOINT} \
  --data '{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
          }
        },
        {
          "type": "text",
          "text": "<image>\nConvert the document to text."
        }
      ]
    }
  ],
  "model": "deepseek-ai/DeepSeek-OCR-2"
}'
```

**Response**

```
{
  "id": "chatcmpl-af755c4e61de3a2a",
  "object": "chat.completion",
  "created": 1773856302,
  "model": "deepseek-ai/DeepSeek-OCR-2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "CINNAMON SUGAR\n\n17,000\n\nGRANDTOTAL\n\nCASH IDR\n\nCHANGEDUE",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 265,
    "total_tokens": 292,
    "completion_tokens": 27,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "kv_transfer_params": null
}
```
