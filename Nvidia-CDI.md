## Issue
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error
- Containers losing access to GPUs with error : "Failed to initialize NVML : Unknown error"

### Normal container startup flow:

- Docker starts container.
- The low level runtime (runc) is the one who knows exactly.
        - Which device are attached.
        - which files are mounted.
        - Which Cgroup permissions are granted.
```
Docker -> runc -> Container
```

### With NVIDIA Runtime hook:
- when you use : `docker run --gpus all ...` OR Older `Nvidia container runtime`.

- An extra NVIDIA component runs after `runc` has prepared the container.
```
Docker -> runc -> Container  <- Nvidia Hook adds:
                                    - GPU devices
                                    - CUDA libraries
                                    - Cgroup permissions
```

The problem is :
- `runc` doesnt know these extra GPU changes we made.
- Suppose docker updates the container configuration:
        - Restart container
        - Update resource limits
        - Change cgroups.

- Docker asks runc:
        - Apply the container configuration again.

> runc only knows the original configs. It does not kknow about the GPU devices added later by NVIDIA hook, So it may accidently remove them. <br>

> Failed to initialize NVML.

### Temparary solution: 
- Restart the container.

### Troubleshoot checking:
```
systemctl daemon-reload
```
- Sometime it gets removed after this command.

### Solution of this:
Method 1 : use cgroupfs as the cgroup driver for continers, to do this, update the /etc/docker/daemon.json to include:
```
{
  "exec-opts": ["native.cgroupdriver=cgroupfs"]
}
```
Then restart:
```
systemctl restart docker
systemctl daemon-reload
```

## CDI:
Method 2 : CDI - GPU declared before the container starts. <br>

Container device interface. <br>
CDI Specification is automatically generated and updated by systemd service called `nvidia-cdi-refresh` <br>
This service Automatically generates by systemd service at `/var/run/cdi/nvidia.yaml` or `/etc/cdi/nvidia.yaml` when: <br>
        - The nvidia container toolkit is installed or upgraded
        - The Nvidia GPU drivers are installed or upgraded
        - The system is rebooted.

### nvidia-cdi-refresh:
`systemctl status nvidia-cdi-refresh`

### List CDI:
List of available CDI Devices: <br>
`nvidia-ctk cdi list`

Ref link - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html <br>

CDI helps to prevent NVML error:
- Container restart
- Container update
- cgroup update
- Docker daemon reload
- Container reconcillation

## If CDI not exists then,

### How to check ?
- `nvidia-gtk cdi list`
- `cat /etc/cdi/nvidia.yaml`

### Setup:
```
mkdir -p /etc/cdi

# Generate CDI Specification.
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia,.yaml
```

### Run workloads with CDI:

```
docker run --rm -ti --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all \
      ubuntu nvidia-smi -L
```
### Setting the CDI Mode Explicitly
```
sudo nvidia-ctk config --in-place --set nvidia-container-runtime.mode=cdi
```


### Sample docker-compose.yaml

docker-compose.yaml
```
version: '3.8'

services:
  purval_service:
    image: gezp/ubuntu-desktop:22.04-cu12.4.1
    devices:
      - "nvidia.com/gpu=0"                          ## GPU added here
    environment:
      - USER=purval
      - PASSWORD=I@m1337
      - GID=1000
      - UID=1000
      - DOCKER_ALLOW_IPV6_ON_IPV4_INTERFACE=1
    container_name: purval_machine
    hostname: test_purval
    cap_add:
      - SYS_ADMIN
      - NET_ADMIN
      - MKNOD
      - SYS_PTRACE
      - AUDIT_WRITE
      - SYS_RESOURCE
      - SYS_NICE
    security_opt:
      - apparmor=unconfined
      - seccomp=unconfined
    deploy:                                 ## resouces assigned here.
      resources:
        limits:
          memory: 256GB
          cpus: '64.0'

    ports:
      - "28570:22"    # SSH Port

    volumes:
      - /sys/fs/cgroup:/sys/fs/cgroup:rw
      - purval:/home/
    networks:
      - docker_default

    # Uncomment the line below if you want the service to restart automatically
    # restart: unless-stopped

    healthcheck:
      test: ["CMD-SHELL", "nvidia-smi > /tmp/nvidia-smi-health.log 2>&1 || exit 1"]
      interval: 300s
      timeout: 50s
      retries: 5
      start_period: 100s

    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

volumes:
  purval:
  purval_var:
  purval_etc:
  purval_usr:

networks:
  docker_default:
    external: true
    name: docker_default

```


## Troubleshooting:

### Issue : One one server CDI based GPU provision working and on another server it is not working.
- `nvidia-ctk cdi list` Showing the output still docker is not starting even after cdi related changes.

Reason:
- docker is attempting CDI handling but can not find/active the CDI drivers.

Check on both servers:
- docker into | grep -i cdi
- docker-compose version
- nvidia-ctk --version
- dpkg -l | grep -i "nvidia-container-toolkit"
- cat /etc/docker/daemon.json

### Solution: upgrade package:
Upgrade nvidia-container-toolkit to 1.18.x. <br>


Then, restart the container,
```
systemctl restart docker
```

### Why Only `nvidia-ctk cdi list` works ?
because reads the CDI specification file (/etc/cdi/nvidia.com) <br>
It does not tell you whether Docker is capable of consuming that CDI spec. <br>
```
nvidia-cdk cdi list ✅
docker info CDI ❌
```
means: <br>
NVIDIA generated the CDI definations, but Docker is not discovering them.


## Upgrade nvidia-container-toolkit
Add repository:
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
```

List avaialble versions:
```
apt-cache madison nvidia-container-toolkit
```
upgrade:
```
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.1-1

sudo apt install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

reconfigure docker:
```
sudo nvidia-ctk runtime configure --runtime=docker
```

Restart docker:
```
sudo systemctl restart docker
```

Regenerate CDI Specs:
```
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```
verify:
```
nvidia-ctk --version
```
