**Library for LLM inference and serving.** <br>
Seamless integration with huggingface models. <br>

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

### Install vLLM with Pip:

1. Create virtual environment and activate it.
```
source ~/.vnev/bin/activate
```

2. install
```
pip install vllm
```

### Start inferencing or serving.
```
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000 \
  --tensor-parallel-size 1
```

If you want faster inference on GPUs like H100/H200.
```
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192
```

### Query the model using API.
```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
          {"role": "user", "content": "Who is sachin tendulkar?"}
        ]
      }'
```
