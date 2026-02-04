
Nvidia AI Stack overview
-----
- Harware : GPUs, CPUs, DPUs, NVLink, NVSwitch
- Software : CUDA, cuDNN, Drivers
- LKibraries/SDK : TensorRT, NCCL, RAPIDS
- AI Frameworks: PyTorch, TensorFlow, JAX
- Tools: Triton, NGC, DeepStream, K8s


Nvidia Software stack:
---
| Key | Value |
| --- | --- |
| CUDA | Software library interact with GPU directly |
| cuDNN | CUDA Deep Neural Network library |
| TensorRT | Inference |

- All are tightly integrated with AI
- Powers training and Inference

- CNN : Convolutional Neural Network <br> Detect Patterns. Object detection in security cameras.
- Tritorn - Supports multi-framework model deploymenbt



Nvidia GPU Cloud (NGC):
--------------
- Registry for GPU containers
- Optimized
- Pre-trained containers
- Secured production ready assets

Nvidia GPU Hardware and software architecture:
---
- SMs - Core building blocks of GPU
        - Handle thread execution and compute operation
        - A100 GPU - has 108 SMs, H100 Has 132 SMs.
        - Each SMs Runs threads and parallalism

- TensorCores - Built for deep learning
        - Accelarate F16, BF16, INT8, TF32, FP8
        - Used in training + Inference
        - Specialized cores for matrix

  SMs + Tensor Cores Accelarate AI
- NVLink - High Speed GPU into connect faster than PCIs.
        - Used in DGX and Suprtcomputer
        - Supports multi-model and multi-GPU.


- Powerful datacenter GPUS - A100, H100, L40s, B200

- A100 : General purpose AI Powerhouse
        - In Most of the provides this is the default choice,
- H100 : Extreeme performance
        - Hopper architecture
        - FP8 Supports
        - LLM Training, GPU Workloads.
- L40S : Idle for enterprise inference + Graphics
        - Ada Lovelance Architecture
        - AI + Graphics + Video
- B200 : Designed for future scale Gen AI.
        - Blackwell architecture

MIG 
--
- Multi instance GPU


DCGM
---
- Monitoring tool
- Detects bottlenecks early
- Prometheus + Grafana


Slurm - Mostly used in GPU job scheduling

Infrastructure Stack - Storage, Networking and Virtualization
------
### GPU Accelarated Storage - Direct data path
- Generally CPU process and align tasks and provides to GPU. and GPU loads into memory. but for the large datasets its not enough. `CPU -> Memory - GPU`
- For large datasets - latency will be high.
- GPU Accelarated storage bypassthe CPU and loads into GPU memory directly. PCIs and NVLink

- GPUDirectStorage (GDS)
- NVMe Over Fabrics
- NVLink interconnects
- DALI
- Compatible with H100, A100

### Networking - infiband vs. Ethernet
- Ethernet is slow
- 1G to 400G speed
- Low Latency
- Packet loss

Infiband:
- 100G-800G Transfer
- Ultra-Low Latency
- RDMA - Zero-Copy Memory access
- Supercomputers

## RDMA
- Data directly trasnfer to GPU memory
- Without CPU passing
- Reduce Latency
- Often used with Infiniband.



## Below are some best inference runtimes:

If you want **raw speed** → **TensorRT-LLM**. <br>
If you want **easy, stable APIs** → **vLLM or NIM**. <br>
If you want **scalable production** → **Triton**. <br>
If you want **local/offline** → **llama.cpp** / **Ollama**. <br>


Below is the typical architecture in terms of working with Nvidia AI Stack:
---------------------------------------------------
Training (NeMo Framework) -> Optimization (TensorRT) -> serving (Triton/NIM)  <br>


### Big Picture:

| Layer | Tool / Framework | Mainly Used By | Purpose |
|-------|------------------|----------------|----------|
| 🧩 **Model Development** | **NeMo Framework** | Data Scientists / ML Researchers | Build, train, and fine-tune AI models (LLMs, ASR, etc.) |
| 🚀 **Model Packaging & Serving** | **NIM (NVIDIA Inference Microservices)** | MLOps Engineers / AI Engineers | Deploy models as scalable microservices (APIs) |
| ⚙️ **Inference Optimization** | **TensorRT / TensorRT-LLM** | MLOps / System Engineers | Optimize model performance for fast GPU inference |
| 🖥️ **Serving Infrastructure** | **Triton Inference Server** | MLOps / DevOps Engineers | Host and serve multiple models efficiently |
| 🧰 **Monitoring / Scaling** | **Kubernetes, Helm, ArgoCD** | MLOps / Platform Engineers | Manage and scale model deployments |


There are 5 main types of LLM model files you will see:
---------------
| Format                  | File Example                              | What it is                      | Where It Comes From         | Can Run On                                                                               |
| ----------------------- | ----------------------------------------- | ------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| **PyTorch Checkpoints** | `pytorch_model.bin` / `model.safetensors` | Raw weights                     | Hugging Face, Meta releases | **vLLM**, **PyTorch**, **Transformers**, **Triton**, **TensorRT-LLM (after conversion)** |
| **GGUF**                | `model.Q4_K_M.gguf`                       | Quantized CPU/GPU format        | llama.cpp community         | **llama.cpp**, **Ollama**, **koboldcpp**, **LM Studio**                                  |
| **ONNX**                | `model.onnx`                              | Framework-agnostic graph format | Export tools, ONNX Runtime  | **ONNX Runtime**, **Triton Inference Server**                                            |
| **TensorRT Engines**    | `model.plan`                              | Optimized GPU execution engine  | **TensorRT-LLM build** step | **TensorRT Runtime**, **NIM**, **Triton Server**, **trtllm-infer**                       |
| **safetensors**         | `model-0001-of-0002.safetensors`          | Safe, memory-mapped HF weights  | Hugging Face                | Same as PyTorch: **vLLM**, **Transformers**, etc.                                        |


### Which runtime supports which format?

| Runtime                     | Supports Safetensors | Supports GGUF |     Supports ONNX     | Supports TensorRT Engine | Notes                          |
| --------------------------- | :------------------: | :-----------: | :-------------------: | :----------------------: | ------------------------------ |
| **vLLM**                    |         ✅ Yes        |      ❌ No     |       ❌ Limited       |           ❌ No           | Best for fast server inference |
| **PyTorch / Transformers**  |         ✅ Yes        |      ❌ No     | ✅ Yes (via exporters) |           ❌ No           | Training + flexible inference  |
| **TensorRT-LLM**            |   ⚠️ Needs convert   |      ❌ No     |          ❌ No         |           ✅ Yes          | Requires **conversion step**   |
| **NVIDIA NIM**              |   ✅ (auto convert)   |      ❌ No     |  ✅ Yes (some models)  |           ✅ Yes          | Production-grade API server    |
| **Triton Inference Server** |         ✅ Yes        |      ❌ No     |         ✅ Yes         |           ✅ Yes          | Enterprise serving platform    |
| **Ollama**                  |         ❌ No         |     ✅ Yes     |          ❌ No         |           ❌ No           | Simple local inference         |
| **llama.cpp**               |         ❌ No         |     ✅ Yes     |          ❌ No         |           ❌ No           | CPU or small GPU inference     |

### When to use which format?

| Goal                                             | Best Format               | Best Runtime                  |
| ------------------------------------------------ | ------------------------- | ----------------------------- |
| **High performance GPU inference (H100 / A100)** | **TensorRT Engine**       | **NIM, Triton, TensorRT-LLM** |
| **Fast inference on consumer GPUs (3090/4090)**  | **vLLM + safetensors**    | **vLLM**                      |
| **Run on Mac / CPU / small GPU**                 | **GGUF**                  | **Ollama / llama.cpp**        |
| **Fine-tune or train**                           | **PyTorch / safetensors** | **PyTorch / Transformers**    |


Real Flow of deploying model:
--------------------------
When we download a model from Hugging Face, it usually comes in:
```
safetensors / .bin weights + config + tokenizer
```

| Runtime / Serving System               | Model Format Required          | Conversion Needed?                  | Notes                        |
| -------------------------------------- | ------------------------------ | ----------------------------------- | ---------------------------- |
| **PyTorch / HuggingFace Transformers** | safetensors / .bin             | ❌ No                                | Slowest but simplest         |
| **vLLM**                               | safetensors / .bin (HF format) | ❌ No                                | Efficient, fast, easy        |
| **ONNX Runtime**                       | .onnx                          | ✅ Convert → ONNX                    | Usually CPU or GPU inference |
| **TensorRT-LLM**                       | `.plan` Engine                 | ✅ Convert → TRT checkpoint → Engine | Fastest on GPU (H100 / A100) |


### If using vLLM
```
HuggingFace model → serve directly
```

### If using ONNX
```
HuggingFace model → convert to ONNX → serve ONNX
```

### If using TensorRT-LLM
```
HuggingFace model (.safetensors) 
        ↓ convert_checkpoint.py
TensorRT-LLM checkpoint
        ↓ trtllm-build
TensorRT Engine (.plan)
        ↓ trtllm-infer / trtllm-serve / NIM / Triton
```

### Which path should we use ?
| Hardware                                    | Recommended Runtime    |
| ------------------------------------------- | ---------------------- |
| **H100 / A100 GPU server (enterprise)**     | **TensorRT-LLM / NIM** |
| **Single GPU consumer cards (4090 / 4080)** | **vLLM**               |
| **CPU only**                                | **ONNX Runtime**       |
| **Laptop / mobile**                         | **GGUF + llama.cpp**   |



Install Huggingface-cli:
-------------------------
```
apt install python3.10-venv
python3 -m venv ~/hf-venv
source hf-venv/bin/activate
pip install --upgrade huggingface_hub

```

Download model from hf:
```
hf download distilbert/distilbert-base-uncased
```

