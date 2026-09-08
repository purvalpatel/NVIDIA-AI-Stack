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
