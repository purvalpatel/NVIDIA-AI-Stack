## Roadmap for Infrastructure to AI
1. [GPU Hardware Layer](#gpu-hardware-layer)
2. [NVIDIA Driver Stack](#Nvidia-driver-stack)
3. [CUDA toolkit](#cuda-toolkit)
4. [Kubernetes Eco-system](kubernetes-eco-system)
5. AI Frameworks
6. [Model Serving](#model-serving)
7. [Nvidia AI Enterprise](#nvidia-ai-enterprise)
8. [NVIDIA NIM](Nvidia-NIM.md)
9. [NeMo](NeMo.md)
10. TensorRT
11. NCCL
12. [Monitoring](#monitoring)
13. [Storage](#storage)
14. Security
15. [Performance Profiling](#performance-profiling)
16. [Cluster management](#cluster-management)

### GPU Hardware Layer
Understand NVIDIA hardware.
- PCIe vs SXM, Tensor Cores, CUDA Cores, NVLink, NVSwitch, Grace CPU, Grace Hopper, H100, H200, B200, RTX Pro GPU
- GPU memory, HBM, Bandwidth, FP32, FP16, BF16, INT8, FP8
- Fore more details [Nvidia GPU Hardware architecture](#nvidia-gpu-hardware-architecture)

### Nvidia Driver stack
- NVIDIA Driver
- NVML
- CUDA Driver API
- CUDA Runtime
- libcuda.so
- nvidia-smi
- persistence mode

### CUDA toolkit
- CUDA Toolkit
- CUDA Runtime
- CUDA Libraries
- CUDA Compiler (nvcc)

### Kubernetes Eco-system
Kubernetes and container eco-system
- NVIDIA Container Toolkit
        - Advanced configuration: Container Device interface (CDI)
- NVIDIA Container Runtime
- [Nvidia CDI](Nvidia-CDI.md)
- OCI Hooks
- Docker Runtime
- Containerd runtime

### Kubernetes and NVIDIA GPUs
- GPU Operator
- Device Plugin
- [GPU Feature Discovery](#gpu-feature-discovery)
- [Node Feature Discovery](#node-feature-discovery)
- [MIG Manager](MIG.md)
- DCGM Exporter
- GPU sharing
- [Time slicing](time-slicing.md)
- [MIG](MIG.md)
- [HAMi](HAMi.md)

### GPU Virtualization
- [MIG](MIG.md)
- [Time Slicing](time-slicing.md)
- [MPS](#mps)
- [vGPU](#vgpu)
- SR-IOV (where supported)
- GPU partitioning

### Model Serving
- [vLLM](vLLM.md)
- [TensorRT-LLM](tensortRT_LLM.md)
- Triton Inference Server
- Ollama
- llama.cpp
- KServe
- SGLang
Know,
- KV Cache
- Tensor Parallelism
- Pipeline Parallelism
- Continuous batching

### Nvidia AI Enterprise
- AI Enterprise
- NIM
- NeMo
- RAPIDS
- Morpheus
- BioNeMo

### Monitoring
- DCGM
- [DCGM Exporter](DCGM-Exporter.md)
- Prometheus
- Grafana
- [Nsight Systems](#nsight-systems)
- [Nsight Compute](#nsight-compute)

### Networking
- NVLink
- NVSwitch
- InfiniBand
- RoCE
- GPUDirect RDMA
- GPUDirect Storage

### Storage
- [GPUDirect Storage](storage.md)
- Lustre
- BeeGFS
- NFS
- NVMe
- Parallel file systems

### Performance profiling
- Nsight Systems
- Nsight Compute
- nvprof (legacy)
- CUPTI

### Cluster management
- Slurm
- Kubernetes
- Volcano
- Kueue
- Ray
- Kubeflow


- PCIs : Socker which connects GPU with the Hardware
- SXM : Socket which is specially designed for super computers to connects the GPU with motherboard.

-----

### Nvidia GPU Cloud (NGC):

- Registry for GPU containers
- Optimized
- Pre-trained containers
- Secured production ready assets


### Nvidia GPU Hardware architecture

- **SMs** - Core building blocks of GPU <br>
        - Handle thread execution and compute operation <br>
        - A100 GPU - has 108 SMs, H100 Has 132 SMs. <br>
        - Each SMs Runs threads and parallalism <br>

- **TensorCores** - Built for deep learning <br>
        - Accelarate F16, BF16, INT8, TF32, FP8 <br>
        - Used in training + Inference <br>
        - Specialized cores for matrix <br>

  SMs + Tensor Cores Accelarate AI <br>
- **NVLink** - High Speed GPU into connect faster than PCIs. <br>
        - Used in DGX and Suprtcomputer <br>
        - Supports multi-model and multi-GPU. <br>


- **Powerful datacenter GPUS** - A100, H100, L40s, B200

- `A100` : General purpose AI Powerhouse <br>
        - In Most of the provides this is the default choice, <br>
- `H100` : Extreeme performance <br>
        - Hopper architecture <br>
        - FP8 Supports <br>
        - LLM Training, GPU Workloads. <br>
- `L40S` : Idle for enterprise inference + Graphics <br>
        - Ada Lovelance Architecture <br>
        - AI + Graphics + Video <br>
- `B200` : Designed for future scale Gen AI. <br>
        - Blackwell architecture <br>

### vGPU
- Software level partitioning
- Hypervisor based
- One physical GPU can be converted into multiple vGPUs and that vGPU can be assigned to VM, User, Containers.

### GPU feature discovery
Kubernetes add-on for detecting GPU features and label it with the node.

### Node feature discovery
Kubernetes add-on for detecting hardware feature and label it with the node.

### MPS
Multi-process service
```
+-----------------------------------------+
|            MPS Server                    |
|                                         |
| Process A  Process B  Process C         |
|      \         |         /              |
|       \        |        /               |
|     Shared CUDA Context                 |
+-----------------------------------------+
```
Instead of each process creating its own CUDA context:
- One MPS Server owns the GPU context.
- Multiple CUDA processes submit work through it.
- The GPU schedules kernels from all processes simultaneously.

### DCGM
- Monitoring tool
- Detects bottlenecks early
- Prometheus + Grafana

### Slurm
**Slurm** - Mostly used in GPU job scheduling

### DPU and Bluefield

#### DPU (Data Processing Unit )
- Specialized processor for data movement tasks.
- Handles networking, storage, security workload to reduce overhead of CPU for this tasks.
- Programmable with SDKs like `DOCA`.

### Nsight Systems 
- Where is my application spending time? (System-wide performance analysis)
- Nsight Systems gives you a timeline view of everything happening on the system.

### Nsight Compute
- Why is this GPU kernel slow? (Individual kernel optimization)
- Nsight Compute focuses on one CUDA kernel.

#### Bluefield
- Nvidia DPU Family
- Runs linux and SDK like `DOCA`
- Used in AI, Edge and HPC infrastructre.

#### DPU offloads:
Move heavy tasks networking/security/storage work from CPU to a dedicated DPU. So CPU can focus on running tasks.
- Network functins: routing, load balancing, firewall
- Storage :  NVMe Access, Compression, Replication
- Security: Encryption, Authentication, Zero-trust enforcement
- Telemetry: Monitoring, Logging, Traffic analysis
Improve GPU throughput.


### Nvidia Eco-system & tools

1. **Nvidia NGC**: Containers, Models, Helm charts
2. **DOCA** SDK and Bluefield DPU usage
3. **Cloud-native** GPU orchestration with K8s <br>
   - Nvidia Container toolkit <br>
   - Device plugins <br>
   - GPU Operator <br>
   - DGCM Exporter <br>
   - NGC Helm chart <br>
  

#### Troubleshooting:
- **nvidia-smi**
- **DCGM**
- **Nvidia Nsight Systems** : Visualize CPU,GPU intercation timelines.
- **Nvidia Nsight Compute** : Kernel level GPU profiling metrics.


   
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



GPU instances on AWS/Zure:
---
A100, H100: <br>
- A100- Ampere architecture
- H100 - Hopper architecture
- Datacenter class, large scale AI training, HPC workloads.

L4: Lightweight, Low-power, used in inferencing <br>

Jetson - Robots, Drone, IoT based devices <br>

AWS: <br>
- p4d/p5 : A100/H100
- g5 : L4

Auzre: <br>
- Nc-series - General purpose
- Nd-series - V100, A100
- Nv-series - Graphic based, not for AI


### DGX Systems:
- AI supercomputers
- Pre-optimized for tensorFlow, RAPIDS

## Nvidia TAO
- Train, Adapt, Optimize framework

- Is a framework for customizing vision foundation models for high accuracy and fine-tuning microservices

### Workflow of TAO
- Pick model from NGC
- Prepare your dataset
- Configure training Spec ( hyperparametersm Augmentation )
- Fine-tune model on your dataset
- optimize (pruning, quantization )

## Deepstream:
- GPU accelrated streaming analytics tooolkit for NVIDIA
- Optimized for realtime video and sensor data
- Camera, Traffic monitoring, theft detection, license, numberplat scaning

Device: Jetson Edge device <br>

```
ingest -> Decode -> AI Inference -> postprocess -> Output
```
- Connects to RAPIDS in readltime analysis


### Realtime Pipeline:
- Input : camera, videfile
- Decode & Preprocess
- Inference : Run optimized AI model (TensorRT)
- Post-process : Draw boxes, count objects, filter events
- Output: dashboard, databases, alerts


### RAPIDS
- Build on CUDA library

## Nvidia Omniverse
- Nvidia's Robotics Simulation platform

Infra SDK's
--------
1. **NVIDIA Metropolish**
- For smart cities
- Components : Deepstream, TAO Toolkit, Triton, RAPIDS

2. **NVIDIA RIVA**
- For Speech recognisation
- Text to speech


3. **NVIDIA Meno for NLP**
- FRamework for training and deploying custom LLMs.

4.** NVIDIA clara**
- For Healthcare AI


5. **NVIDIA Merlin**
- Recommendation system


Some Common used Terms in Nvidia AI Stack
---------------------
| Key | Value | 
| -- | -- |
| GPU | GPU Processing Unit |
| DPU | Data Processing unit |
| NVLink | Connects GPU-to-GPU communication | 
| CUDA | Python library to use GPUs |
| cUDANN | CUDA Deep Neural Network library |
| TensorRT | optimized inference engine |
| NCCL | a specialized software library that enables high-speed, low-latency data transfer and communication between multiple NVIDIA GPUs and nodes |
| RAPIDS | an open-source suite of GPU-accelerated libraries and APIs designed to speed up end-to-end data science, analytics, and machine learning pipelines | 
| PyTorch | AI Framework | 
| TensortFlow | an open-source machine learning framework that is heavily optimized to run on NVIDIA GPUs | 
| JAX | a high-performance numerical computing and machine learning library that leverages NVIDIA GPUs for massive acceleration |
| ONNX | Multi framework Model file |
| Triton | Inference server | 
| NGC | Nvidia container registry | 
| DeepStream | real-time video, audio, and image analytics | 
| SMs | Building block of GPU where threads are running | 
| TensorCores | hardware unit for matrix multiplication  | 
| MIG | Hardware level partitioning |  
| vGPUs | Virtual GPU. Software level paritioning | 
|  DGCM | Monitoring tool by nvidia | 
| Slurm | High compute | 
| GPUDirectStorage | transfer data directly to GPU memory | 
| DALI | Data Offloading library to overcome CPU bottlenecks | 
| NVMe | High speed storage protocols that  connects SSDs to CPUS directly with PCIs | 
| InfiniBand | GPU to GPU communication channel | 
| RDMA | Remote direct memory Access, used in GPUDirectStorage | 
| BlueField | Nvidia DPU family | 
| DOCA | Library for DPUs | 
| Nvidia Nsignt compute | Kernel level GPU profiling metrics | 
| Nvidia Nsight systems | Visualize CPU-GPU interaactions | 


<img width="1231" height="1635" alt="Untitled Diagram drawio(4)" src="https://github.com/user-attachments/assets/918b8526-c0a5-46ca-bb91-420eeb029261" />

