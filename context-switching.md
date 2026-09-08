## Context Switching vs. MPS

### Context Switching
it is the process of switching GPU execution state from one CUDA context to another.

CUDA context is like GPU execution environment:
1. CUDA streams
2. Memory mappings
3. CUDA runtime state
4. GPU allocations

Context Swithching = Take Turns <br>
                      GPU Switches between different programs.


### MPS = share the GPU

Allows multiple CUDA processes to send work through an MPS server.

NVIDIA Technology that allows multiple CUDA processes to share a GPU with less context-switching overhead.

## Content Moderation
Checking the content and deciding whether it is safe and acceptable according to certain rules.

```
"Message containing prohibited content"
        ↓
      UNSAFE ❌
        ↓
    Block / Flag
```

in AI/LLM system, content moderation often used before and After an LLM.
```
User Input
    ↓
Content Moderation
    ↓
   LLM
    ↓
Output Moderation
    ↓
User
```
For example, an AI chatbot might check:

Input → Is the user's request allowed? <br>
LLM processing → Generate response. <br>
Output → Is the generated response safe? <br>
Return response → Send it to the user. <br>


Process of automatically or manually reviewing user-generated or AI-generated content to identify and filter content that violates policies.

### tool for Content moderation prevention
1. Nvidia Nemo Guardrails
2. Custom ML models
3. OpenAI Moderation.
