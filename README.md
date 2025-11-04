Tensor is and ecosystem of tools for developers to achieve high performance deep learning inference. <br>

**TensorRT includes:**
- Inference compilers(TensorRT Compiler),  
- TensorRT-LLM,
- Runtimes,
- model optimizations(TensorRT Model Optimizor) ,
- TensortRT-RTX ( For RTX GPUs used for Laptops ),
- TensortRT-Cloud. 
<br>
that delivers **low latency** and **high throughput** in production applications. 

 
Get Started with TensorRT :
-------------------------

TensorRT is developer tool used to: <br>

1. convert models ( From pyTorch, tensorFlow, ONNS) into an optimized GPU format 
2. Deploy those models in production 
3. Speed up inference. 

### Use case: 
1. **Inference for LLMs** - (Data center GPUs ) 
2. **Inference for non-LLMs** like CNNs, difusions, Transformers, etc. ( Data center GPUs ) 
3. **Safety-complaint** and **high-performance inference** for automotive embedded 
4. Inference for non-LLMs in robotics and edge applications. 
5. AI model inferencing on RTX PCs. 
6. Model optimizations like Quantization, Distillation, sparsity, etc. ( Data center GPUs) 

### Nvidia TensorRT Model Optimizer 

Optimization techniques includes quantization, distillation, pruning, speculative decoding and sparsiry to accelarate models. 

**Techniques:** <br>
1. Post Training Quantization
2. Quantiation Aware Training.
3. Pruning
4. Distillation
5. Speculative Decoding
6. Sparsity 

 
**Typical workflow:** <br>
1. Train model - (PyTorch , TensorFlow). 
2. Export to ONNX Format. 
3. Convert ONNX -> TensorRT engine. 
4. Run inference using TensorRT runtime or NVIDIA Triton inference server. 


**Step 1:** Install tensorRT 
```BASH
pip install tensorrt 
pip install nvidia-pyindex 
pip install onnx onnxruntime 
```
 

**Step 2:** Export Model to ONNX <br>
Write pytorch model. <br>
This creates an **ONNX** flie (**resnet50.onnx**) - a universal model format.  <br>


**Step 3:** Convert ONNX -> TensorRT Engine  <br>
Use tensorRT tool trtexec: <br>
```
trtexec --onnx=resnet50.onnx --saveEngine=resnet50.engine --fp16 
```
 
**Step 4:** Run inference <br>
```
trtexec --loadEngine=resnet50.engine --shapes=input:1x3x224x224 
```
 
**Step 5:** Use Triton Inference server ( Optional ) <br>

This is used for model runtime. So we can query model. <br>
```
docker run --gpus all -p8000:8000 -v /models:/models nvcr.io/nvidia/tritonserver:24.10-py3 \ 
  tritonserver --model-repository=/models 
```
 

### TensorRT-LLM 
- Optimized engine to run LLMs on NVIDIA GPUs. 
- Library for runtime and optimizing LLM inference on NVIDIA GPUs. 
- Built on the top of the existing TensorRT stack, but specialized for LLMs. 

**Why use ?** <br>
- Reduce latency and increases throughput, lower cost on GPU infrastructure. 
- Efficient batching, memory usage, GPU utilization. 

 

#### Typical workflow: 
1. Get a model  <br>
    e.g. LLaMA, Mistral ( Weights in .safetensorts format ) 

2. Convert using TensorRT-LLM tools:  <br>
```
trtllm-build --model_dir ./TinyLlama --output_dir ./engine --dtype float16 
```
3. Run it with TensorRT-LLM runtime(trtllm-serv) or NVIDIA triton runtime server:  <br>
```
trtllm-serve ./engine 
```
3. You now have an API endpoint (like openAI’s API) that runs the model at GPU-optimized speed.  <br>

 

**Full architecture:  <br>**
```
model.safetensors  →  TensorRT-LLM build  →  model.engine  →  fast inference 
```
 
- Safetensors -> Stores the model weights safely, it is just a data not executable file. 
- TensorRT-LLM -> Run those model extremly fast on Nvidia GPUs. 

 

**Real-world example:**

1. Download model  <br>

Download model from the hugging face (like, LLaMMa, Mistral etc. )  <br>
Model files will be like, .safetensors, config.json, tokenizer.json  <br>

2. Convert to TensorRT-LLM  <br>

TensorRT-LLM reads the .safetensorts weights and converts them into a GPU-optimized binary engine.  <br>
```
trtllm-build --model_dir ./mistral --output_dir ./engine --dtype float16 
```
3. run the model.  <br>
Use TensorRT-LLM runtime or NVIDIA Triton inference server to host it.  <br>
```
trtllm-serve ./engine 
```
 
Now you can query it like:  <br>
```
curl -X POST http://localhost:8000/v1/completions -d '{"prompt":"Hello!"}'
```

Together they are used in production grade AI inference – the same tech which is behind the chatgpt.  <br>

Nvidia NIM :
----------
- NVIDIA NIM is a collection of pre-built, optimized microservice containers that packages
```
AI Models + Inference runtime (TensorRT-LLM + REST API ) + GPU Optimizations.
```

It packages, <br>

**AI Models + Inference Runtime(TensorRT-LLM or Tritorn inference server ) + GPU Optimizations.**  <br>

- NVIDIA NIM for LLM supports Multi-instance GPU (MIG) Mode. MIG is best of smaller parameter models. ( less then 8 billion )  <br>

You can download NIM containers directly from,  <br>

https://catalog.ngc.nvidia.com/search?orderBy=scoreDESC&query=nim

- NIM can save lot of setup time.  <br>
- You dont need to built from scratch like optimization part.  <br>


**How NIM works ?** <br>

**Step 1.** You select a model that supported by NIM.  <br>

To pull NIM image from NGC, first authenticate Nvidia container registry:  <br>
```
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```
You can download and run the NIM of your choice using one of three approaches:  <br>

#### 1. API Catelog  <br>

It’s a cloud platform where NVIDIA hosts and runs the models for you. Means we can question the model and all processing is done at nvidia cloud platform side.  <br>

https://build.nvidia.com/  <br>

 

#### 2. NGC:

You can run images by pulling it from the Nvidia cloud.  <br>

For the you have to setup NGC CLI.  <br>

https://org.ngc.nvidia.com/setup/installers/cli  <br>

https://catalog.ngc.nvidia.com/  <br>

List NIM supported image list with CLI:  <br>
```
./ngc registry image list --format_type csv nvcr.io/nim/* 
```
 

#### 3. HuggingFace or Localdisk:  <br>

If an LLM-specific NIM container is not yet available for your model, you can use the multi-LLM compatible NIM container to deploy own model. 

 

The process is same as NGC deployments but with different launch process that is only supported by the multi-LLM compatible NIM container. 

You can set NIM_MODEL_NAME to download from the check points. 

https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#option-1-deploy-from-api-catalog-llm-specific-nim 

 

**Step 2.** Deploy NIM container (Docker / Kubernetes ) with GPU.  <br>

Container contains ( APIs+Inference Enginer + Optimized kernels ) 

**Step 3.** Container runtime inspects your hardware (GPU, Memory ) and picks optimal inference enginer or model variant. 

 

**Step 4.** Once Running  <br>

Applications (Frontend, chatbot) Sends requests (prompts) to NIM Microservice which generates response using optimized GPU runtime. 

**Step 5.** You just monitor, scale , update microservice as needed.  <br>

 

**Note**:  <br>

- Nvidia NIM can run custom models also.  <br>
- You dont need to run inference enginer seperately.  <br>
- NIM already includes the inference runtime inside. Which is ( TensorRT-LLM, Triton inference runtime )  <br>

 

NeMo ( For Developers ) :
----------------------

Provides pre-built containers that already have all the AI developement tools you need – GPU Optimized.  <br>

NeMo Provides tools for 3 main things:  <br>
1. Training models  <br>
2. Customizing models  <br>
3. Deploying models  <br>

 
Three main building blocks of NeMo:  <br>
1. **Nemo Framework**  - For training and customizing models  <br>
2. **NeMo Guardrails**	- Adds safety and rules to chatbots  ( For Developers )  <br>
3. **Nemo Inference**	- For deploying optimized models ( For MLOps )  <br>

 

#### NeMo Framework: <br>
Its an open-source framework to train, fine-tune and deploy large AI Models – like chatbots, speech.  <br>
Toolkit to build and customize LLM and generative models efficiently.  <br>

**Why NeMo?**  <br>
Before NeMo, Developers had to manually handle:  <br>
Distributed training  <br>
GPU Parallesim  <br>
Data preprocessing  <br>
Checkpoint management  <br>
Optimization for mixed precission(FP16,FP8)  <br>
NeMo Automates all that for NVIDIA hardware.  <br>

https://github.com/NVIDIA-NeMo/NeMo 

#### Example in real-life:  <br>

1. Start with NVIDIA’s pre-trained LLM in NeMo.  <br>
2. Fine-tune it using your hospitals document.  <br>\
3. Use NeMo tools to deploy it securely and make it respond safely.  <br>

Example: Create customer supoort chatbot using NeMo.  <br>

**Step 1:** Start with the pre-trained model.  <br>
```
ngc registry model download-version nvidia/nemo:1.22.0 
```
**Step 2:** Fine-tune it on your own data. 

You prepare your bank FAQs or documents as text files: 
```
data/  <br> 
├── faq_loan.txt  <br>
├── faq_creditcard.txt  <br> 
└── faq_account.txt   <br>
```
 

**Step 3:** Add Safety with NeMo Guardrails 

Import nemoguardrails libraries. <br>
You dont want your chatbot to talk about unrelated things or leaked info. <br>

**Step 4:** Deploy using **Nvidia Triton** + **TensorRT** 


Big Picture:
----------
| Layer | Tool / Framework | Mainly Used By | Purpose |
|-------|------------------|----------------|----------|
| 🧩 **Model Development** | **NeMo Framework** | Data Scientists / ML Researchers | Build, train, and fine-tune AI models (LLMs, ASR, etc.) |
| 🚀 **Model Packaging & Serving** | **NIM (NVIDIA Inference Microservices)** | MLOps Engineers / AI Engineers | Deploy models as scalable microservices (APIs) |
| ⚙️ **Inference Optimization** | **TensorRT / TensorRT-LLM** | MLOps / System Engineers | Optimize model performance for fast GPU inference |
| 🖥️ **Serving Infrastructure** | **Triton Inference Server** | MLOps / DevOps Engineers | Host and serve multiple models efficiently |
| 🧰 **Monitoring / Scaling** | **Kubernetes, Helm, ArgoCD** | MLOps / Platform Engineers | Manage and scale model deployments |
