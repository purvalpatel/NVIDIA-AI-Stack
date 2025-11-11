**Library for LLM inference and serving.** <br>
Seamless integration with huggingface models. <br>

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

### Install vLLM with Pip:

1. Create virtual environment and activate it.
```BASH
source ~/.vnev/bin/activate
```

2. install
```BASH
pip install vllm
```

### Start inferencing or serving.
```BASH
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000 \
  --tensor-parallel-size 1
```

If you want faster inference on GPUs like H100/H200.
```BASH
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192
```

### Query the model using API.
```BASH
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
          {"role": "user", "content": "Who is sachin tendulkar?"}
        ]
      }'
```

Another example:
---------------
1. Download small model from huggingface.
```BASH
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
2. Create virtual environment with python3.10:
```BASH
python3.10 -m venv vllm_env
source vllm_env/bin/activate
```
Install pytorch with cuda12:
```BASH
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm
```

3. Serve it with vLLM.
```BASH
vllm serve /home/nuvo_admin/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6 --host 0.0.0.0 --port 8000
```

4. Test model
This is chat model. but we can test this in below way. this may not work. 
```BASH
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama-1.1B-Chat-v1.0",
    "prompt": "Hello"
  }'

```
