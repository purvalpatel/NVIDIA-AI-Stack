NVIDIA Fabric Manager (FM) is a Linux service used on certain NVIDIA multi-GPU servers to manage and monitor the NVSwitch/NVLink fabric that connects GPUs.

On supported NVIDIA **HGX/DGX-type** systems, it helps initialize and manage the NVLink/NVSwitch fabric.

where GPUs need very high-bandwidth communication for:

- Tensor parallelism
- Distributed LLM inference
- Training
- NCCL
- CUDA multi-GPU applications

The NVIDIA driver handles normal GPU functionality.

Fabric Manager handles the management/initialization of the NVSwitch fabric on systems that require it.

check status:
```
systemctl status nvidia-fabricmanager

systemctl is-enabled nvidia-fabricmanager
```

