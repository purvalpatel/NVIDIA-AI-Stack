
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
