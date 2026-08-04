## Infrastructure Stack - Storage, Networking and Virtualization

### GPU Accelarated Storage - Direct data path
Generally CPU process and align tasks and provides to GPU and GPU loads into memory. but, for the large datasets its not enough. <br>
`CPU -> Memory - GPU` <br>
For large datasets - latency will be high in this scenarios. <br>
GPU Accelarated storage bypass the CPU and loads into GPU memory directly. <br>


- `GPUDirectStorage` (GDS)
- `NVMe` Over Fabrics
- `NVLink` interconnects
- `DALI`
Compatible with H100, A100

#### Networking - infiniband vs. Ethernet
- `Ethernet` is slow
- 1G to 400G speed
- Low Latency
- Packet loss

while `Infiband`:
- 100G-800G Transfer
- Ultra-Low Latency
- RDMA - Zero-Copy Memory access
- Supercomputers

### RDMA and Direct Storage
#### RDMA:
- Data directly trasnfer to GPU memory
- Without CPU passing
- Reduce Latency
- Often used with Infiniband.

#### Direct Storage access GPU
- Extends RDMA Concepts
- Allows NVMe drives, NFS servers can transfer data directly to GPU memory.
- This is possible with `CUDA`, `cFile`, `DALI`
- Supported A100, H100 and blueField

#### Why RDMA and Direct Storage Matter ?
- Speed up training and inference
- Involves GPU utilization
- Reduce GPU load and cost
- Supports large batch streaming and multi-node clusters.
- Enable Realtime Data ingetion
