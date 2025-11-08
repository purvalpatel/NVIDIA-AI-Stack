Deploy TensorRT-LLM checkpoints & engines with NIM:
---------------------------------------------------

This includes: <br>
- Take Large Language Model (**LLM**) -> **convert** to **TensorRT-LLM** checkpoiubts
- **Deploy** TensorRT-LLM checkpoints with NIM
- **Compile** TensorRT-LLM Engine
- Deploy **TensorRT-LLM** engine with **NIM**
- **Test** **TensorRT-LLM** engine deployment.

system setup:
```
!nvidia-smi
```

Install required softwares:
```
%pip install docker requests huggingface-hub && echo "✓ Python dependencies installed successfully"
```

```
import os
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
```

Nvidia NGC API Keys:
```

import getpass
import os

if not os.environ.get("NGC_API_KEY", "").startswith("nvapi-"):
    ngc_api_key = getpass.getpass("Enter your NGC API Key: ")
    assert ngc_api_key.startswith("nvapi-"), "Not a valid key"
    os.environ["NGC_API_KEY"] = ngc_api_key
    print("✓ NGC API Key set successfully"
```
Docker login:
```
!echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Huggingface token:
```

if not os.environ.get("HF_TOKEN", "").startswith("hf_"):
    hf_token = getpass.getpass("Enter your Huggingface Token: ")
    assert hf_token.startswith("hf_"), "Not a valid key"
    os.environ["HF_TOKEN"] = hf_token
    print("✓ Hugging Face token set successfully")
```

### Setup NIM Container:

```
# Set the NIM image
os.environ['NIM_IMAGE'] = "nvcr.io/nim/nvidia/llm-nim:latest"
print(f"Using NIM image: {os.environ['NIM_IMAGE']}")

```
Pull NIM container image:
```
!docker pull $NIM_IMAGE && echo "✓ NIM container image pulled successfully"
```

Below is the testing of NIM in notebook environment, not required in general:
```
import requests
import time
import docker
import os

def check_service_ready_from_logs(container_name, print_logs=False, timeout=600):
    """
    Check if NIM service is ready by monitoring Docker logs for 'Application startup complete' message.

    Args:
        container_name (str): Name of the Docker container
        print_logs (bool): Whether to print logs while monitoring (default: False)
        timeout (int): Maximum time to wait in seconds (default: 600)

    Returns:
        bool: True if service is ready, False if timeout reached
    """
    print("Waiting for NIM service to start...")
    start_time = time.time()

    try:
        client = docker.from_env()
        container = client.containers.get(container_name)

        # Stream logs in real-time using the blocking generator
        log_buffer = ""
        for log_chunk in container.logs(stdout=True, stderr=True, follow=True, stream=True):
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"❌ Timeout reached ({timeout}s). Service may not have started properly.")
                return False

            # Decode chunk and add to buffer
            chunk = log_chunk.decode('utf-8', errors='ignore')
            log_buffer += chunk

            # Process complete lines
            while '\n' in log_buffer:
                line, log_buffer = log_buffer.split('\n', 1)
                line = line.strip()

                if print_logs and line:
                    print(f"[LOG] {line}")

                # Check for startup complete message
                if "Application startup complete" in line:
                    print("✓ Application startup complete! Service is ready.")
                    return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    print(f"❌ Timeout reached ({timeout}s). Service may not have started properly.")
    return False

def check_service_ready():
    """Fallback health check using HTTP endpoint"""
    url = 'http://localhost:8000/v1/health/ready'
    print("Checking service health endpoint...")

    while True:
        try:
            response = requests.get(url, headers={'accept': 'application/json'})
            if response.status_code == 200 and response.json().get("message") == "Service is ready.":
                print("✓ Service ready!")
                break
        except requests.ConnectionError:
            pass
        print("⏳ Still starting...")
        time.sleep(30)

def generate_text(model, prompt, max_tokens=1000, temperature=0.7):
    """Generate text using the NIM service"""
    try:
        response = requests.post(
            f"http://localhost:8000/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

print("✓ Utility functions loaded successfully")
```

### Download base model:

```
# Set base directory for all files - you can modify this path as needed
# Examples: ".", "~", "/tmp", "/scratch", etc.
base_work_dir = "/ephemeral"
os.environ["BASE_WORK_DIR"] = base_work_dir

# Set up model download location
model_save_location = os.path.join(base_work_dir, "models")

os.environ["MODEL_SAVE_LOCATION"] = model_save_location
os.environ["LOCAL_MODEL_DIR"] = os.path.join(model_save_location, "llama3-8b-instruct-hf")

# Create model directory
os.makedirs(os.environ["LOCAL_MODEL_DIR"], exist_ok=True)
```
If model is like LLaMA we need to download from huggingface then we need to generate requests.<br>
```
!huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir "$LOCAL_MODEL_DIR" && echo "✓ Model downloaded successfully"
```


```
!echo "@GSEKaFl9_nRN" | sudo -S docker run --rm \
  --runtime=nvidia \
  --gpus '"device=0,1"' \
  --shm-size=16GB \
  -v /home/nuvo_admin/.purval/ephemeral/models/llama3-8b-instruct-hf/original:/home/nuvo_admin/.purval/input_model -v TRTLLM_CKPT_DIR:/home/nuvo_admin_/.purval/output_model \
  -u $(id -u) \
  $NIM_IMAGE \
  python3 app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
  --model_dir /input_model \
  --output_dir /output_dir/trtllm_ckpt \
  --dtype bfloat16
```
