Deploy Huggingface Safetensor LLM Weight with NIM:
-------------------------------------------

Quick summary:

1. Take model from hugging face.
2. Prepare local folder for cache : will mount this cache folder with container because if we not do this it will download everytime and takes time.
3. run NVIDIA NIM container to serv that model.

**It is for serving model, not for training.**

Prerequisites:
1. Docker
2. Recent NIM Docker image
3. Huggingface login and Access key
4. Local cache directory for NIM
5. Nvidia developer access key


### First login docker for nvcr.io
```
docker login nvcr.io
username - $oauthtoken
password - TOKEN
```

### Setup nvidia container toolkit if not installed.
```
nvidia-container-tookit
```

## Deployment Example 1:
Run docker command with NIM container

```
!docker run -it --rm \
 --name=$CONTAINER_NAME \
 --runtime=nvidia \
 --gpus all \
 --shm-size=16GB \
 -e HF_TOKEN=$HF_TOKEN \
 -e NIM_MODEL_NAME="hf://mistralai/Codestral-22B-v0.1" \
 -e NIM_SERVED_MODEL_NAME="mistralai/Codestral-22B-v0.1" \
 -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
 -u $(id -u) \
 -p 8000:8000 \
 -d \
 $NIM_IMAGE
```



## Deployment Example 2:
Deployment using different backend options.
1. TensorRT-LLM

```

# Using TensorRT-LLM backend by specifying the NIM_MODEL_PROFILE parameter
!docker run -it --rm \
 --name=$CONTAINER_NAME \
 --runtime=nvidia \
 --gpus all \
 --shm-size=16GB \
 -e HF_TOKEN=$HF_TOKEN \
 -e NIM_MODEL_NAME="hf://mistralai/Codestral-22B-v0.1" \
 -e NIM_SERVED_MODEL_NAME="mistralai/Codestral-22B-v0.1" \
 -e NIM_MODEL_PROFILE="tensorrt_llm" \
 -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
 -u $(id -u) \
 -p 8000:8000 \
 -d \
 $NIM_IMAGE
```

2. vLLM
```
# Using vLLM backend by specifying the NIM_MODEL_PROFILE parameter
docker run -it --rm \
 --name=$CONTAINER_NAME \
 --runtime=nvidia \
 --gpus all \
 --shm-size=16GB \
 -e HF_TOKEN=$HF_TOKEN \
 -e NIM_MODEL_NAME="hf://mistralai/Codestral-22B-v0.1" \
 -e NIM_SERVED_MODEL_NAME="mistralai/Codestral-22B-v0.1" \
 -e NIM_MODEL_PROFILE="vllm" \
 -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
 -u $(id -u) \
 -p 8000:8000 \
 -d \
 $NIM_IMAGE
```



## Deployment Example 3:
Customizing different parameters. <br>
Key parameters: <br>
`NIM_TENSOR_PARALLEL_SIZE=2`: Uses 2 GPUs in parallel for better performance
`NIM_MAX_INPUT_LENGTH=2048`: Limits input to 2048 tokens
`NIM_MAX_OUTPUT_LENGTH=512`: Limits output to 512 tokens

```
!docker run -it --rm \
 --name=$CONTAINER_NAME \
 --runtime=nvidia \
 --gpus all \
 --shm-size=16GB \
 -e HF_TOKEN=$HF_TOKEN \
 -e NIM_MODEL_NAME="hf://mistralai/Codestral-22B-v0.1" \
 -e NIM_SERVED_MODEL_NAME="mistralai/Codestral-22B-v0.1" \
 -e NIM_TENSOR_PARALLEL_SIZE=2 \
 -e NIM_MAX_INPUT_LENGTH=2048 \
 -e NIM_MAX_OUTPUT_LENGTH=512 \
 -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
 -u $(id -u) \
 -p 8000:8000 \
 -d \
 $NIM_IMAGE
```

## Deployment Example 4:
Depployment with local model, run your own model which is available in local system. <br>

We'll download Qwen2.5-0.5B, a lightweight LLM, for use in Example 4. <br>

```Notebook

# Set up local model directory
model_save_location = os.path.join(base_work_dir, "models")
local_model_name = "Qwen2.5-0.5B-Instruct"
local_model_path = os.path.join(model_save_location, local_model_name)
os.makedirs(local_model_path, exist_ok=True)

os.environ["LOCAL_MODEL_DIR"] = local_model_path
```

```
!huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$LOCAL_MODEL_DIR" && echo "✓ Model downloaded successfully"
```

```
!docker run -it --rm \
 --name=$CONTAINER_NAME \
 --runtime=nvidia \
 --gpus '"device=0"' \
 --shm-size=16GB \
 -e NIM_MODEL_NAME="/opt/models/local_model" \
 -e NIM_SERVED_MODEL_NAME="Qwen/Qwen2.5-0.5B" \
 -v "$LOCAL_MODEL_DIR:/opt/models/local_model" \
 -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
 -u $(id -u) \
 -p 8000:8000 \
 -d \
 $NIM_IMAGE
     
```

### Test model deployment:
```
check_service_ready_from_logs(os.environ["CONTAINER_NAME"], print_logs=True)
```

### Check with prompt:
```
result = generate_text(model="Qwen/Qwen2.5-0.5B",
                       prompt="Tell me a story about a cat")
print(result if result else "Failed to generate text")
```
