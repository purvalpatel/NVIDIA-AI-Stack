NVIDIA GPU Time-Slicing is a software-based GPU sharing technique where multiple workloads share the same physical GPU by taking turns using the GPU for short time intervals. Unlike MIG, time-slicing does not partition the GPU hardware—all workloads share the full GPU.

### How Time-Slicing Works
```
                    Physical GPU
                +------------------+
                | 80 GB HBM Memory |
                | 132 SMs          |
                +------------------+
                        │
          ┌─────────────┼─────────────┐
          │             │             │
       Pod A         Pod B         Pod C
          │             │             │
          └─────────────┼─────────────┘
                        │
                Time-Slicing Scheduler
                        │
      ┌─────────────────┼─────────────────┐
      │                 │                 │
    5 ms              5 ms              5 ms
      │                 │                 │
    Pod A             Pod B             Pod C

```

The GPU scheduler rapidly switches between workloads, giving each one a small time quantum (typically a few milliseconds). Because the switching is very fast, all workloads appear to run concurrently.

### Kubernetes Flow
```
Pod A (GPU)
        │
Pod B (GPU)
        │
Pod C (GPU)
        │
        ▼
NVIDIA Device Plugin
        │
        ▼
NVIDIA Time-Slicing
        │
        ▼
GPU Driver Scheduler
        │
        ▼
Physical GPU
```

The NVIDIA Device Plugin advertises virtual GPU slots instead of the actual number of GPUs.

For example, if you have 1 GPU and configure:
```
replicas: 10
```
the node may advertise:
```
Capacity:
  nvidia.com/gpu: 10
```
Although there is only one physical GPU, Kubernetes believes there are 10 schedulable GPU resources.

Example Configuration

With the NVIDIA Device Plugin:
```
version: v1
sharing:
  timeSlicing:
    resources:
    - name: nvidia.com/gpu
      replicas: 10
```
Now:
```
kubectl describe node
```
shows:
```
Capacity:
  nvidia.com/gpu: 10
```

### Scheduling Example

Suppose you have one H100 GPU.

Without time-slicing:
```
GPU
 │
 └── Pod A
```

Only one Pod can request:
```
resources:
  limits:
    nvidia.com/gpu: 1
```
With time-slicing (10 replicas):
```
GPU

├── Pod A
├── Pod B
├── Pod C
├── Pod D
├── ...
└── Pod J
```
All ten Pods request:
```
resources:
  limits:
    nvidia.com/gpu: 1
```
but each gets a time-shared slice of execution, not a dedicated GPU.

### GPU Memory

This is the biggest limitation.

Time-slicing does not partition GPU memory.

If the GPU has:
```
80 GB HBM
```
then all Pods share that memory.

Example:
```
Pod A → 20 GB

Pod B → 25 GB

Pod C → 15 GB

Total = 60 GB
```
This works.

But:
```
Pod A → 40 GB

Pod B → 30 GB

Pod C → 25 GB

Total = 95 GB
```
The third allocation will fail with an out-of-memory (OOM) error because memory is not isolated.