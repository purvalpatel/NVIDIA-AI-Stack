# llama.cpp

Llama.cpp is an open-source C/C++ inference engine designed to run Large Language Models (LLMs) efficiently on CPUs and GPUs.

It support GGUF Model format.

ollama internally uses llama.cpp inference engine.

## Deploy kimi-2.6 model with llama.cpp [Quantized Model-Q4]

### Create Virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

### install dependencies
```
pip install huggingface-hub
pip install huggingface-cli
```

### Login Huggingface
```
hf auth login
```

Dry run of download
```
hf download bartowski/moonshotai_Kimi-K2.6-GGUF --dry-run
```

### kimi k2.6 Q4 download
```
hf download bartowski/moonshotai_Kimi-K2.6-GGUF \
>   --include "moonshotai_Kimi-K2.6-Q4_0/*" \
>   --local-dir ./Kimi-K2.6-Q4_0

```

## Once Download start llama-cpp server with docker-compose.
Dockerfile
```
FROM rocm/dev-ubuntu-22.04:6.4

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    hipblas-dev \
    rocblas-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/ggml-org/llama.cpp.git

WORKDIR /app/llama.cpp

#RUN git checkout b3407

RUN cmake -B build \
    -DGGML_HIP=ON \
    -DCMAKE_C_COMPILER=hipcc \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DAMDGPU_TARGETS=gfx942 \
    -DCMAKE_BUILD_TYPE=Release

RUN cmake --build build -j 4
#RUN cmake --build build --target llama-server --verbose -j 4

EXPOSE 8080

ENTRYPOINT ["./build/bin/llama-server"]
```

docker-compose.yaml
```
services:
  kimi:
    build: .

    container_name: kimi-server

    restart: unless-stopped

    devices:
      - /dev/kfd
      - /dev/dri

    group_add:
      - video
      - render

    security_opt:
      - seccomp=unconfined

    environment:
      - HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      - HSA_OVERRIDE_GFX_VERSION=11.0.0

    ports:
      - "8080:8080"

    volumes:
      - /Data/models/Kimi-k2.6-GUFF/BF16:/models
    ulimits:
      memlock: -1
      stack: 67108864

    command: >
      -m /models/Kimi-K2.6-BF16-00001-of-00046.gguf
      --host 0.0.0.0
      --port 8080
      -ngl 999
      -c 16384
      -t 32
      --parallel 8
      --mlock
      --tensor-split 1,0,0,0,0,0,1,1
```

Another docker-compose.yaml for Image `llama.cpp:server-rocm` [ this is working ]
```
services:
  llama-server:
    image: ghcr.io/ggml-org/llama.cpp:server-rocm
    container_name: llama-server

    ports:
      - "8079:8080"

    volumes:
      - /Data/hf_cache/unsloth/Kimi-K2.6-GGUF:/models

    devices:
      - /dev/kfd
      - /dev/dri

    group_add:
      - video
      - render

    command:
      - -m
      - /models/UD-Q8_K_XL/Kimi-K2.6-UD-Q8_K_XL-00001-of-00014.gguf
      - --no-warmup
      - --host
      - 0.0.0.0
      - --port
      - "8080"
      - -n
      - "512"

    restart: unless-stopped
```

Here i have AMD MI300X GPU with 4 cards.
