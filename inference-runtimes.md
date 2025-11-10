AI Inference Runtimes (Low-Level, Fast Execution):
----------------------------------------------------

| Runtime                                           | Hardware                 | Notes                                                         |
| ------------------------------------------------- | ------------------------ | ------------------------------------------------------------- |
| **TensorRT / TensorRT-LLM**                       | NVIDIA GPUs              | Fastest inference for NVIDIA. Requires compiling engines.     |
| **ONNX Runtime**                                  | CPU + GPU (multi-vendor) | General inference runtime; flexible; used by many frameworks. |
| **GGUF + llama.cpp**                              | CPU, GPU (low VRAM)      | Very lightweight, great for local and edge.                   |
| **FasterTransformer** (deprecated → TensorRT-LLM) | NVIDIA GPUs              | Old NVIDIA optimized transformer runtime.                     |
| **OpenVINO**                                      | Intel CPU/GPU            | Best for Intel-based inference optimizations.                 |
| **AMD ROCm + MIGraphX**                           | AMD GPUs                 | AMD’s inference stack; similar to TensorRT but less mature.   |


High-Performance LLM Serving Engines:
---------------------------------
| Tool                                            | Built On                                  | Strength                                                            |
| ----------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------- |
| **vLLM**                                        | CUDA + PagedAttention                     | **Fast high-throughput LLM serving**, best for chat and batching.   |
| **Triton LLM (NVIDIA Triton Inference Server)** | TensorRT + multiple backends              | **Enterprise serving**, autoscaling, metrics, multi-model serving.  |
| **NIM (NVIDIA Inference Microservices)**        | Triton + TensorRT-LLM + pre-built engines | **Production API out-of-the-box**, easiest to deploy.               |
| **TensorRT-LLM Serve (`trtllm-serve`)**         | TensorRT-LLM                              | **Lightweight API**, lower overhead than Triton, not full-featured. |
| **Text Generation Inference (TGI)**             | PyTorch                                   | Hugging Face’s serving stack; simple and stable.                    |
| **DeepSpeed-MII / DeepSpeed-Inference**         | CUDA kernels                              | Optimized PyTorch inference for large LLMs.                         |


Open-Source Local Runtimes (Desktop / Private / Offline):
-------------------------------------
| Tool                          | Best Use Case                                |
| ----------------------------- | -------------------------------------------- |
| **Ollama**                    | Easiest local LLM runner with chat UI / CLI. |
| **LM Studio**                 | No-config GUI for Windows/Mac models.        |
| **Jan (formerly LocalAI UI)** | Lightweight desktop UI.                      |
| **KoboldCpp / KoboldAI**      | Local text-gen with RP/creative UIs.         |


Inference Middleware / Distributed Serving:
--------------------------------
| Tool           | Purpose                                          |
| -------------- | ------------------------------------------------ |
| **Ray + vLLM** | Scale vLLM to multiple machines.                 |
| **SGLang**     | Fast instruction serving + sequence parallelism. |
| **LightLLM**   | High efficiency inference without TensorRT.      |




Understanding Where Each Fits:
-------------------

| Layer                        | Examples                              | Purpose                          |
| ---------------------------- | ------------------------------------- | -------------------------------- |
| **Low-level inference core** | TensorRT-LLM, ONNX Runtime, llama.cpp | Runs model math efficiently      |
| **Serving engine**           | vLLM, TGI, Triton, NIM                | Handles batching, streaming, API |
| **User app / UI layer**      | Ollama, Chat UI, API client           | Interface users interact with    |


