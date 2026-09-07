If you want to implement MPS, the basic Idea is:
```
              NVIDIA GPU
                  │
             MPS Control
                  │
          ┌───────┼───────┐
          │       │       │
        App-1   App-2   App-3
          │       │       │
          └────── CUDA ────┘
```

The most use of MPS is multiple CUDA applications are using the one GPU.

### check MPS is available or not?
```
nvidia-smi
```
then,
```
which nvidia-cuda-mps-control
which nvidia-cuda-mps-server
```
You should normally have these from the CUDA toolkit.

### check CUDA version:
```
nvcc --version
```

### Start MPS daemon
Create MPS directory:
```
sudo mkdir -p /var/run/nvidia-mps
sudo chmod 777 /var/run/nvidia-mps
```
Set the directory:
```
export CUDA_MPS_PIPE_DIRECTORY=/var/run/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
```

Create the log directory:
```
sudo mkdir -p /var/log/nvidia-mps
sudo chmod 777 /var/log/nvidia-mps
```
Start MPS:
```
nvidia-cuda-mps-control -d
```
Verify:
```
ps -ef | grep mps

### Run CUDA applications
```
