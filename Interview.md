Round 1 – AI/HPC Scaling, K8s, Cloud & Linux 
--------------
1. How would you auto-scale GPU nodes for training workloads without wasting GPU hours on idle pods?
2. A multi-cluster, multi-region AI training job fails halfway because one cluster runs out of GPU memory. How do you rebalance workloads live?
3. How do you configure Kubernetes taints and tolerations for GPU workloads?
4. How would you handle CUDA driver upgrades in K8s without disrupting thousands of running AI pods?
5. Explain how you’d pre-warm GPU nodes for massive AI inference traffic (e.g., ChatGPT-scale) with zero cold-start penalty.
6. How would you monitor GPU utilization in real-time in a Kubernetes cluster?

Round 2 – RCA, Fire Drills & GPU Chaos
--------------
1. How would you check if a GPU pod in Kubernetes is using the GPU assigned to it?
2. What are NCCL logs, and why are they important in distributed training?
3. Persistent storage for AI datasets starts showing 200ms+ latency. How do you pinpoint whether it’s the storage backend, the network, or the GPU node?
4. A Kubernetes GPU pod requests 16GB VRAM but only gets 12GB due to fragmentation. How do you detect and fix in real-time?
5. Your AI pipeline cost doubles in 24 hours with no infra change. Profiling shows a silent GPU resource leak. How do you hunt it down?

Round 3 – Leadership, Reliability Culture & Scaling Influence 
--------------
1. How do you set up SLOs for both AI inference latency and batch training completion times without overprovisioning GPUs?
2. You’re told to implement multi-region AI inference failover without DNS-based routing. What’s your plan?
3. How do you justify infra cost for idle GPU pre-warming to leadership when each hour costs $30–$40 per GPU?
