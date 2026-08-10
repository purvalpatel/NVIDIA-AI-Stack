## issue 1:
```Log
[2026-08-10 10:37:28] MDGP054:3306732:3306732 [4] transport/nvls.cc:284 NCCL WARN Failed to bind NVLink SHARP (NVLS) Multicast memory of size 2097152 : CUDA error 1 'invalid argument'.
This is usually caused by a system or configuration error in the Fabric Manager or NVSwitches.
Disable NVLS (NCCL_NVLS_ENABLE=0) if you wish to avoid this error in the future.
```

### Troubleshooting steps:
This NCCL warning is specifically related to NVLink SHARP / NVLS multicast.

#### What happens internally.
Your NCCL process is trying to use:
```
NCCL
 └── NVLS (NVLink SHARP)
      └── NVLink/NVSwitch multicast
           └── Fabric Manager / NVSwitch
```
#### Check the GPU topology
```
nvidia-smi topo -m
```
<img width="1173" height="211" alt="image" src="https://github.com/user-attachments/assets/93f2aab0-b6c0-462c-8d21-69e0c9e7f930" />

Also,
```
nvidia-smi nvlink -s
```
If still not works then reboot the server.
