
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
2. Fine-tune it using your hospitals document.  <br>
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

#### NeMo Guardrails 

Is a safety and control system for AI applications. <br>

#### Nemo Inference: 

The system that turns your trained model into a fast, scalable, production-ready service optimized for NVIDIA GPUs. <br>
