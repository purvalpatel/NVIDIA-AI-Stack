## 1. What is NVIDIA MIG?

MIG (Multi-Instance GPU) is a feature available on NVIDIA Ampere (A100) and Hopper (H100) GPUs. <br>
It lets you partition a single physical GPU into multiple isolated GPU instances, each with dedicated compute cores, memory, and bandwidth.

### Benefits:

- Multiple users or workloads can share a single GPU without interfering with each other.
- Perfect for serving multiple models or workloads simultaneously.
- Each MIG instance behaves like a smaller GPU.

## 2. Requirements

- GPU: NVIDIA A100, H100 (MIG supported GPUs)
- NVIDIA Driver: 470+ (for A100), 525+ (for H100)
- CUDA Toolkit installed
- nvidia-smi tool available

### Check GPU and driver:
```
nvidia-smi
```
This will show the GPU, driver version, and if MIG is supported.

## 3. Enable MIG Mode

Check current MIG mode:
```
nvidia-smi -L
```
### First check it is enabled or not?
```
nvidia-smi -i 0
nvidia-smi -i 0 --query-gpu=pci.bus_id,mig.mode.current --format=csv
```
<img width="933" height="706" alt="Screenshot from 2026-03-07 12-31-13" src="https://github.com/user-attachments/assets/9834a5c5-bb25-4c2c-adfe-111e3c3c8e9e" />

If you see “GPU 0: NVIDIA H100 …” with no MIG instances, MIG is not enabled.

### Enable MIG on the GPU:
```
sudo nvidia-smi -i 0 -mig 1
```
After enabling, the GPU will restart (no processes should be running on it). <br>

### Verify MIG mode:
```
nvidia-smi
nvidia-smi -L
```
Now you should see a `MIG-enabled` GPU, often listed as multiple “GPU Instances.” <br>

### List all possible MIG instances. 
```
nvidia-smi mig -lgip
## for specific GPU - 0
nvidia-smi mig -lgip -i 0
```
<img width="554" height="415" alt="image" src="https://github.com/user-attachments/assets/d4d5876e-1bb6-4ee4-a42a-51b4fda99636" />

## 4. Create MIG Instances
You can create multiple GPU instances (GI) with different sizes. Each instance has a compute instance (CI). <br>

### Example: create 3 instances of H100 with size 1g.10gb:
```
sudo nvidia-smi mig -i 0 -cgi 19,19,19 -C
OR
nvidia-smi mig -cgi 9 -i 0 -C
```
`-i 0` → GPU 0 <br>
`-cgi` → Compute GPU Instance profile ID <br>
`-C` → Create <br>

### List available profiles:
```
nvidia-smi mig -lgip
```

This will show profile IDs like 1g.5gb, 2g.10gb, etc. You choose based on your workload. <br>

## 5. Verify MIG Instances
```
nvidia-smi
nvidia-smi -L
```

You should now see a list of GPU Instances, each with its memory, cores, and utilization. <br>

Example output snippet: <br>
```
GPU 0 (H100) MIG Mode: Enabled
  GI 0 1g.10gb
  GI 1 1g.10gb
  GI 2 1g.10gb
```

## 6. Use MIG Instances in CUDA / PyTorch / vLLM

Each MIG instance is exposed as a separate GPU device. <br>

In PyTorch:
```
import torch
# List all visible GPUs including MIG instances
print(torch.cuda.device_count())
# Select specific MIG instance
device = torch.device("cuda:0")
```

In vLLM or TensorRT, use `CUDA_VISIBLE_DEVICES` to select a MIG instance:

```
CUDA_VISIBLE_DEVICES=0 vllm serve <model_name> ...
```

Here 0 refers to the first MIG instance.

## 7. Delete / Reset MIG Instances

To delete all MIG instances:
```
sudo nvidia-smi mig -i 0 -dci all
```
`-dci` all → Delete all compute instances on GPU 0 <br>

To fully reset MIG:
```
sudo nvidia-smi mig -i 0 -R
```
Destroy all CI and GI:
```
nvidia-smi mig -dci && sudo nvidia-smi mig -dgi
```

## NVIDIA MIG-Parted

When we create GPU partitions and reboot the server, the partitions GPU will be automatically removed and when we create it the UUID will be changed. <br>
To overcome this issue. We have to use nvidia-mig-parted tool. <br>

### Install MIG-PARTED
Install [nvidia-mig-parted](https://github.com/NVIDIA/mig-parted/releases) Download deb file and install it. <br>

Clone the mig-parted git repository:
```
cd /home/script
git clone https://github.com/purvalpatel/mig-parted.git
```

Now create/edit config YAML file for the configuration inside `/path/to/mig-parted/mig-parted/examples/config.yaml`

Location on live server: /path/to/mig-parted/

#### config.yaml
```
version: v1
mig-configs:
  - devices: [0]
    mig-enabled: true
    mig-devices:
      1c.3g.71gb: 1
      1g.18gb: 4
      2c.3g.71gb: 1
  - devices: [1, 2, 3, 4, 5, 6]
    mig-enabled: false
    mig-devices: {}
  - devices: [7]
    mig-enabled: true
    mig-devices:
      1c.3g.71gb: 1
      1g.18gb: 3
      2c.3g.71gb: 1
```

Verify the configuration are proper or not.
```
nvidia-mig-parted assert -f config.yaml
```
Apply the changes:
```
nvidia-mig-parted apply -f config.yaml
```

Verify it is working fine or not?
```
reboot
```

After reboot apply below command.
```
nvidia-mig-parted apply -f config.yaml
```
The same partitions will be created.

Note: UUID will change sometimes even after reboot. MIG-parted will make sure to recreate all the MIG Partition after reboot. but not give suriety of UUID stay intact. instead of Using UUID in program you can use CDI id of the GPU device. <br>
<br>
for example, `docker run --rm-it \   --device=nvidia.com/gpu=0:4 \   ubuntu:22.04 bash`

<br>
You will get this cdi list using
```
nvidia-ctk cdi list
```
<img width="573" height="865" alt="image" src="https://github.com/user-attachments/assets/291d04ea-20bb-40e3-b214-5682d1fc4944" />

### Disable MIG of specific GPU.
Remove all MIG instances from GPU 1:
```
sudo nvidia-smi mig -i 1 -dci
sudo nvidia-smi mig -i 1 -dgi
```
Disable MIG Mode of GPU 1:
```
sudo nvidia-smi -i 1 -mig 0
```
Verify:
```
nvidia-smi -L
```
Here, -i 1, is GPU 1.
