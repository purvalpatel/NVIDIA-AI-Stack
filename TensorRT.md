
Tensor is and ecosystem of tools for developers to achieve high performance deep learning inference. <br>

**TensorRT includes:**
- Inference compilers(TensorRT Compiler),  
- TensorRT-LLM,
- Runtimes,
- model optimizations(TensorRT Model Optimizor) ,
- TensortRT-RTX ( For RTX GPUs used for Laptops ),
- TensortRT-Cloud. 
<br>
that delivers low latency and high throughput in production applications. 

 
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
Create Python virtual environment. <br>
```BASH
pip install tensorrt 
pip install nvidia-pyindex 
pip install onnx onnxruntime
pip install tensorrt_llm
```
 

**Step 2:** Export Model to ONNX <br>
- Write pytorch model. <br>
- This creates an **ONNX** flie (**resnet50.onnx**) - a universal model format.  <br>


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
```
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir "$LOCAL_MODEL_DIR" && echo "✓ Model downloaded successfully"
```

3. Convert using TensorRT-LLM tools:  <br>
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
Note: <br>
- trtllm-build command will not work properly without using container because pre-created container will contain all the dependencies into container.

```
docker run -it --rm \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16G \
  -v /home/nuvo_admin:/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2
```
Then inside the container:
```
cd /workspace/.purval/output_model/trtllm_ckpt
trtllm-build --checkpoint_dir . --output_dir ./engine --dtype float16
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


 




