# HAMi – NVIDIA GPU Virtualization on Kubernetes

> A practical guide to fractional and whole-GPU sharing in a Kubernetes cluster using
> HAMi (Heterogeneous AI Computing Virtualization Middleware).

---

## Table of Contents

1. [What is HAMi? (Full Description)](#1-what-is-hami-full-description)
2. [HAMi Architecture](#2-hami-architecture)
3. [Purpose (in terms of virtualization)](#3-purpose-in-terms-of-virtualization)
4. [Objective (in terms of virtualization)](#4-objective-in-terms-of-virtualization)
5. [Installation](#5-installation)
6. [Test Cases with Explanation](#6-test-cases-with-explanation)
7. [Implementation](#7-implementation)
8. [Appendix: Quick Reference](#appendix-quick-reference)

---

## 1. What is HAMi? (Full Description)

**HAMi** — *Heterogeneous AI Computing Virtualization Middleware* (formerly known as
*k8s-vGPU-scheduler*) — is an open-source, **CNCF Sandbox** project that provides GPU
(and other accelerator) virtualization and sharing for Kubernetes. It lets a single
physical GPU be safely divided among multiple pods, and lets the cluster schedule those
pods with awareness of each device's real memory and compute capacity.

### The problem it addresses

Kubernetes, on its own, has no concept of a *partial* GPU. The default NVIDIA device
plugin advertises whole GPUs only, so the smallest possible unit a pod can request is
one entire card. On expensive, high-memory accelerators this is extremely wasteful:
a workload that needs 3 GB still locks an 80 GB card, and other pods queue behind it
even though the hardware is mostly idle. HAMi introduces a middleware layer that makes
GPUs behave like a divisible, schedulable pool of memory and compute.

### What HAMi provides

- **Fine-grained device sharing** — split one physical GPU into many virtual GPUs,
  each bounded by a specific amount of memory (in MiB or as a percentage) and a
  percentage of compute cores.
- **Hard isolation** — a container only sees and can only use the resources it was
  granted. Attempts to exceed the memory quota fail with an out-of-memory error
  instead of impacting co-located pods.
- **Whole-device allocation** — full GPUs, or several full GPUs, can still be handed
  to a single pod when a workload genuinely needs them.
- **Device-aware scheduling** — a scheduler extender places pods based on real device
  inventory, using policies such as *binpack* (consolidate) or *spread* (distribute).
- **Heterogeneous hardware support** — beyond NVIDIA, the project supports a range of
  accelerators (e.g. certain AMD, Cambricon, Hygon, Iluvatar, MThreads, Ascend
  devices), presenting a unified sharing model across vendors.
- **Transparency to applications** — no code changes, no custom CUDA builds, and no
  special image are required; workloads continue to use the standard
  `nvidia.com/...` resource requests.

### How it fits into the cluster

HAMi installs alongside (and replaces the role of) the standard NVIDIA device plugin.
It registers virtual GPU resources with the kubelet, intercepts GPU pod scheduling
through a webhook and scheduler extender, and injects a lightweight interception
library into workload containers to enforce limits at the CUDA API level. From an
operator's point of view, it is a Helm-installed add-on; from a user's point of view,
GPU requests simply gain new, finer-grained fields.

### When to use it

HAMi is a strong fit for shared clusters running a mix of inference services,
notebooks, development workloads, and batch jobs — anywhere many GPU consumers must
coexist efficiently on limited hardware. It is equally capable of serving large
training jobs that need one or more whole GPUs, making it suitable as the single GPU
management layer for a heterogeneous cluster.

---

## 2. HAMi Architecture

HAMi is built from cooperating components that span the Kubernetes control plane, each
GPU node, and the inside of every GPU workload container.

### 2.1 Component overview

| Component | Runs where | Responsibility |
|-----------|-----------|----------------|
| **Mutating admission webhook** | Control plane | Intercepts GPU pod creation and marks the pod to be scheduled by the HAMi scheduler instead of the default one. |
| **Scheduler extender** | Control plane | Filters and scores nodes based on real GPU inventory, selects a physical GPU that can fit the request, and writes the chosen device UUID into the pod's annotations during bind. |
| **Device plugin (HAMi)** | Each GPU node | Registers *virtual* GPU resources (`nvidia.com/gpu`, `nvidia.com/gpumem`, `nvidia.com/gpucores`) with the kubelet; advertised count = physical GPUs × split count. |
| **HAMi-core / `libvgpu.so`** | Inside each workload container | Intercepts CUDA API calls (via `LD_PRELOAD`) and enforces the per-pod memory and compute limits, so the container only sees its slice. |
| **Node inventory annotations** | Each GPU node object | `hami.io/node-nvidia-register` records each GPU's UUID, index, split count, total memory, and compute — the data the scheduler reads. |

### 2.2 Text architecture diagram

```
                        ┌─────────────────────────────────────────────┐
                        │              Kubernetes Control Plane         │
                        │                                               │
   kubectl apply pod    │   ┌───────────────┐      ┌─────────────────┐  │
   ──────────────────►  │   │  API Server   │─────►│ HAMi Mutating   │  │
                        │   │               │      │ Webhook         │  │
                        │   │               │◄─────│ (routes to HAMi)│  │
                        │   └──────┬────────┘      └─────────────────┘  │
                        │          │                                    │
                        │          ▼                                    │
                        │   ┌─────────────────────┐                     │
                        │   │  HAMi Scheduler      │  reads node GPU     │
                        │   │  Extender            │  inventory, picks   │
                        │   │  (filter + score +   │  a physical GPU,    │
                        │   │   bind)              │  writes UUID to     │
                        │   └──────────┬──────────┘  pod annotation      │
                        └──────────────┼──────────────────────────────┬─┘
                                       │ bind pod to node             │
                                       ▼                              │
        ┌──────────────────────────────────────────────────────────┐ │
        │                       GPU Node                            │ │
        │                                                           │ │
        │   ┌────────────────────┐        ┌──────────────────────┐  │ │
        │   │  Kubelet           │◄───────│ HAMi Device Plugin   │  │ │
        │   │                    │ registers  (advertises vGPUs: │  │ │
        │   │                    │  vGPUs   physical × split)     │  │ │
        │   └─────────┬──────────┘        └──────────────────────┘  │ │
        │             │ starts container                            │ │
        │             ▼                                             │ │
        │   ┌──────────────────────────────────────────────────┐   │ │
        │   │  Workload Container                               │   │ │
        │   │   ┌───────────────────────────┐                   │   │ │
        │   │   │ Application (CUDA)         │                   │   │ │
        │   │   └───────────┬───────────────┘                   │   │ │
        │   │               │ CUDA API calls                    │   │ │
        │   │               ▼                                   │   │ │
        │   │   ┌───────────────────────────┐   enforces mem /  │   │ │
        │   │   │ HAMi-core (libvgpu.so)     │   compute limits  │   │ │
        │   │   │ injected via LD_PRELOAD    │                   │   │ │
        │   │   └───────────┬───────────────┘                   │   │ │
        │   └───────────────┼───────────────────────────────────┘   │ │
        │                   ▼                                       │ │
        │            ┌─────────────┐   ┌─────────────┐  ...         │ │
        │            │ Physical GPU│   │ Physical GPU│              │ │
        │            │  (shared by │   │             │              │ │
        │            │  many pods) │   │             │              │ │
        │            └─────────────┘   └─────────────┘              │ │
        └───────────────────────────────────────────────────────────┘ │
                                                                       │
        Node object annotation:  hami.io/node-nvidia-register ◄────────┘
        (per-GPU UUID, index, split count, devmem, devcore)
```

### 2.3 End-to-end flow (what happens when a GPU pod is created)

1. **Submit** — a user applies a pod requesting, say, `nvidia.com/gpu: 1` and
   `nvidia.com/gpumem: 5120`.
2. **Webhook** — the HAMi mutating webhook intercepts the pod and sets its
   `schedulerName` to the HAMi scheduler.
3. **Schedule** — the scheduler extender reads each node's
   `hami.io/node-nvidia-register` inventory, filters to nodes/GPUs that can fit
   5 GB, scores them (e.g. binpack), and selects a physical GPU.
4. **Bind** — the scheduler writes the chosen physical GPU UUID into the pod's
   `hami.io/vgpu-devices-allocated` annotation and binds the pod to the node.
5. **Device plugin** — the node's HAMi device plugin provides the container with
   access to the selected physical GPU and passes the allocation details.
6. **Inject** — HAMi-core (`libvgpu.so`) is loaded into the container via
   `LD_PRELOAD`.
7. **Enforce** — at runtime, HAMi-core intercepts CUDA calls; the container sees only
   5 GB of GPU memory, and allocations beyond the quota fail. Other pods on the same
   physical GPU are unaffected.

### 2.4 Design principles reflected in the architecture

- **Control-plane vs data-plane separation** — scheduling decisions happen centrally
  (webhook + extender), while enforcement happens locally in each container
  (HAMi-core). This keeps the scheduler simple and the isolation close to the
  workload.
- **Annotation-driven state** — device inventory and per-pod allocation are stored as
  Kubernetes annotations, making the system observable with plain `kubectl` and
  avoiding hidden external state.
- **API-level interception, not virtualization of hardware** — for the software
  sharing path, isolation is enforced by intercepting CUDA calls rather than by
  hardware partitioning, which is why it works on cards that do not support hardware
  MIG.

---

## 3. Purpose (in terms of virtualization)

In a standard Kubernetes cluster, the NVIDIA device plugin treats a GPU as an
**indivisible unit**. When a pod requests `nvidia.com/gpu: 1`, the scheduler binds
the *entire* physical card to that pod — regardless of whether the workload uses
2 GB or 60 GB of memory, or 5% or 100% of the compute.

This creates two chronic problems on any shared GPU cluster:

- **Under-utilization** – small inference jobs, notebooks, and dev workloads lock a
  full high-memory card (e.g. an 80 GB H100) while using a fraction of it. The rest
  of the card sits idle but unschedulable.
- **Artificial scarcity** – because each GPU can serve only one pod at a time, new
  pods stay in `Pending` state waiting for a whole card to free up, even though there
  is plenty of unused GPU memory and compute across the cluster.

**GPU virtualization** solves this by inserting a virtualization layer between
Kubernetes and the physical device. Instead of exposing "1 GPU = 1 unit," the layer
exposes each physical GPU as **multiple logical (virtual) GPUs**, each carved to a
specific slice of memory and compute. Multiple pods can then share one physical card,
each isolated so it only "sees" the resources it requested.

The purpose of HAMi in this context is to provide that virtualization layer for
NVIDIA GPUs (and other accelerators) **without requiring any change to application
code** — workloads keep using standard Kubernetes resource requests.

---

## 4. Objective (in terms of virtualization)

The virtualization objectives HAMi is designed to meet:

| Objective | What it means |
|-----------|---------------|
| **Device sharing** | Allocate a *fraction* of a physical GPU — by memory, by compute cores, or by device count — so many pods run on one card. |
| **Resource isolation** | Enforce per-pod hard limits on GPU memory and compute. A pod requesting 5 GB sees exactly 5 GB; allocation beyond its quota returns an out-of-memory error rather than stealing from neighbors. |
| **Whole-device allocation** | Still support handing a full GPU — or several full GPUs — to a single pod when a workload genuinely needs it (e.g. large-model training). |
| **Device-aware scheduling** | Place pods intelligently across the cluster using policies such as *binpack* (pack cards tightly) or *spread* (distribute evenly). |
| **Zero application change** | Workloads use the same `nvidia.com/gpu` request pattern; virtualization is transparent to the container. |
| **Higher utilization** | Convert idle GPU capacity into schedulable capacity, raising effective cluster throughput without buying more hardware. |

### How the virtualization works (conceptually)

HAMi is composed of a few cooperating pieces:

- **Mutating webhook** – intercepts GPU pod creation and routes the pod to HAMi's
  scheduler.
- **Scheduler extender** – reads each node's GPU inventory (UUID, total memory,
  compute) from node annotations and picks a physical GPU that can fit the request.
  It writes the chosen physical GPU UUID into the pod's annotations during the *bind*
  phase.
- **Device plugin** – registers the *virtual* GPU count with the kubelet. After
  install, `nvidia.com/gpu` on a node reflects **physical GPUs × split count**
  (default split = 10).
- **In-container isolation library (HAMi-core / `libvgpu.so`)** – injected into the
  workload container. It intercepts CUDA API calls and enforces the memory/compute
  limits, so each container only sees its allocated slice.

Key virtualization semantics to remember:

- After install, the node's advertised `nvidia.com/gpu` = number of **vGPUs**.
- In a **pod request**, `nvidia.com/gpu` still means the number of **physical GPUs**
  the pod needs.
- `nvidia.com/gpumem` (MiB) or `nvidia.com/gpumem-percentage` slices the memory
  **per physical GPU**.
- Overcommit is a *logical* view for higher utilization — it does not create
  additional physical memory. A pod cannot use more than a card physically has.

---

## 5. Installation

### 5.1 Prerequisites

- A working Kubernetes cluster with GPU nodes.
- NVIDIA driver installed on each GPU node.
- NVIDIA container runtime configured as the default runtime (if the stock NVIDIA
  device plugin already exposes GPUs, this is already satisfied).
- Helm installed and configured against the cluster.

### 5.2 Important pre-checks

**A. Remove the stock NVIDIA device plugin.**
HAMi ships its own device plugin that also registers `nvidia.com/gpu`. The HAMi
plugin and the official NVIDIA plugin **must not coexist** on the same node — both
registering the same resource causes conflicts.

```bash
# find how it was deployed first
helm list -A | grep -i gpu
kubectl -n kube-system get daemonset | grep -i nvidia

# if it is a standalone daemonset, remove it
kubectl -n kube-system delete daemonset <nvidia-device-plugin-daemonset>
```
> If the plugin was installed by the NVIDIA GPU Operator, do **not** delete the
> daemonset directly — instead disable the operator's built-in plugin
> (`devicePlugin.enabled=false`).

**B. Label your GPU nodes.**
HAMi's scheduler only manages nodes labeled `gpu=on`. Without this label the node
cannot be scheduled by HAMi.

```bash
kubectl label node <gpu-node> gpu=on --overwrite
```

**C. Match the scheduler image to your Kubernetes version.**
`scheduler.kubeScheduler.imageTag` must match the cluster's **API-server** version.
A mismatch is the most common cause of a silent scheduling failure.

```bash
kubectl version | grep -i server     # e.g. Server Version: v1.33.x
```

### 5.3 Install with Helm

```bash
helm repo add hami-charts https://project-hami.github.io/HAMi
helm repo update

helm install hami hami-charts/hami \
  --set scheduler.kubeScheduler.imageTag=<vX.YY.Z> \
  -n kube-system
```

Replace `<vX.YY.Z>` with the server version from the pre-check.

### 5.4 Verify the install

```bash
kubectl -n kube-system get pods | grep hami
```

Expected — one device-plugin pod **per GPU node** plus the scheduler, all `Running`:

```
hami-device-plugin-xxxxx   2/2   Running   0   ...
hami-device-plugin-yyyyy   2/2   Running   0   ...
hami-scheduler-zzzzz       2/2   Running   0   ...
```

Confirm the nodes now advertise vGPUs (physical × split count, default ×10):

```bash
kubectl get node <gpu-node> -o jsonpath='{.status.capacity.nvidia\.com/gpu}{"\n"}'
# a node with 8 physical GPUs should now report 80
```

Confirm HAMi inventoried the physical cards:

```bash
kubectl get node <gpu-node> -o yaml | grep -i hami.io/node-nvidia-register
# lists each GPU UUID, index, count (split), devmem (MiB), devcore
```

---

## 6. Test Cases with Explanation

All test pods below use a CUDA base image that contains `nvidia-smi`. This is enough
to **verify allocation** (what memory/how many cards the container sees). To verify
enforcement under real load, a workload that actually allocates GPU memory is needed
(see §7.4).

> Verification principle: the scheduler *running* does not prove isolation. The proof
> is what `nvidia-smi` reports **inside** the container, and which physical GPU UUID
> the pod was bound to.

### 6.1 Fractional sharing – many small pods on one card

**Goal:** show that multiple pods share a single physical GPU by memory.

Each pod requests a fixed memory slice (e.g. 5 GB = 5120 MiB) and one physical GPU to
slice from. With the default split count of 10, up to 10 pods can share one physical
card; additional pods spill to the next card.

```yaml
resources:
  limits:
    nvidia.com/gpu: 1          # one physical GPU to share from
    nvidia.com/gpumem: 5120    # 5 GB visible in this container
```

**Explanation of expected result:** if 15 such pods are launched, they distribute as
~10 on the first physical GPU and ~5 on a second — because the *split count* (10),
not the memory, is the first limit reached. Grouping the pods by their allocated GPU
UUID reveals the sharing (see §7.3).

### 6.2 Case 1 – One whole GPU to one pod (legacy behavior)

**Goal:** reproduce the old "1 pod = 1 full card" behavior.

```yaml
resources:
  limits:
    nvidia.com/gpu: 1          # full memory + full compute, no slice
```

**Explanation:** omitting `gpumem`/`gpucores` means no isolation cap is applied, so
the container sees the **entire** card memory. `nvidia.com/gpu: 1` is kept explicit so
the container is scoped to exactly one card.

### 6.3 Case 2 – Multiple whole GPUs to one pod

**Goal:** give several full physical GPUs to a single pod (e.g. multi-GPU training).

```yaml
resources:
  limits:
    nvidia.com/gpu: 2          # 2 DISTINCT whole physical GPUs
```

**Explanation:** in a pod request, the GPU count means *distinct physical GPUs*. HAMi
will **not** stack multiple vGPUs from the same card onto one pod — so this requires 2
free physical cards on a single node. `nvidia-smi -L` inside the container should list
two cards.

### 6.4 Case 3 – Multiple GPUs, each with a specific memory size

**Goal:** combine multi-GPU allocation with per-card memory slicing.

```yaml
resources:
  limits:
    nvidia.com/gpu: 2          # 2 physical GPUs
    nvidia.com/gpumem: 20480   # 20 GB on EACH of the 2 GPUs (40 GB total)
```

**Explanation:** the `gpumem` value is applied **per physical GPU**, not as a shared
total. For portability across GPU models with different memory (e.g. 80 GB vs 48 GB),
`nvidia.com/gpumem-percentage` can be used instead of an absolute MiB value.

### 6.5 Summary of cases

| Case | `nvidia.com/gpu` | `gpumem` | Result |
|------|------------------|----------|--------|
| Fractional | `1` | small (e.g. 5120) | many pods share one card |
| Case 1 | `1` | omitted | one full physical GPU (legacy) |
| Case 2 | `N` | omitted | N full physical GPUs to one pod |
| Case 3 | `N` | `<MiB>` | N GPUs, that memory on **each** |

---

## 7. Implementation

### 7.1 Requesting a slice (portable form)

Because different GPU models have different total memory, prefer the percentage form
when a single spec must run on mixed hardware:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  schedulerName: hami-scheduler
  containers:
    - name: app
      image: <your-workload-image>
      resources:
        limits:
          nvidia.com/gpu: 1
          nvidia.com/gpumem-percentage: 25   # 25% of whichever card it lands on
          nvidia.com/gpucores: 30            # cap at 30% compute
```

### 7.2 Verifying allocation inside a pod

```bash
# total memory the container sees — should equal the requested slice, not the full card
kubectl exec -it <pod> -- nvidia-smi --query-gpu=memory.total --format=csv

# list the GPUs visible to the container (useful for multi-GPU cases)
kubectl exec -it <pod> -- nvidia-smi -L

# a plain nvidia-smi shows the HAMi-core init line, confirming the isolation library is active
kubectl exec -it <pod> -- nvidia-smi
```

### 7.3 Proving which physical GPU each pod uses

HAMi writes the chosen physical GPU UUID into the pod annotation
`hami.io/vgpu-devices-allocated` (format: `GPU-<uuid>,NVIDIA,<mem>,<cores>:;`).
Pods sharing the same UUID are on the same physical card.

```bash
# count pods per physical GPU (requires jq)
kubectl get pods -l <label-selector> -o json \
  | jq -r '.items[].metadata.annotations["hami.io/vgpu-devices-allocated"]' \
  | cut -d',' -f1 | sort | uniq -c

# per-pod mapping
kubectl get pods -l <label-selector> -o json \
  | jq -r '.items[] | .metadata.name + "  ->  " + (.metadata.annotations["hami.io/vgpu-devices-allocated"] // "none")'
```

Map a UUID back to a physical card on the host:

```bash
nvidia-smi -L      # each line: GPU N: <model> (UUID: GPU-xxxx)
```

> Note: pods that only sleep never open a CUDA context, so **host `nvidia-smi` shows
> no processes and 0% util** even while many pods share the cards. For idle pods, the
> annotation grouping above is the correct proof. Host-level utilization only appears
> when a real GPU workload runs.

### 7.4 Verifying enforcement under real load

To confirm the memory cap actually holds, run a workload that tries to allocate more
than its quota. A pod limited to a small slice should hit an out-of-memory error at
its quota rather than reaching the full card size — this is the true test of
isolation, beyond what `nvidia-smi` reporting shows.

### 7.5 Scheduling policy for mixed workloads

Running whole-GPU jobs (Case 1/2) and fractional jobs together can fragment cards — a
card already hosting small slices can't be handed out "whole" until they finish. To
keep whole cards available:

- Set the default scheduling policy to **binpack** so fractional pods pack tightly and
  leave more whole cards free.

  ```yaml
  scheduler:
    defaultSchedulerPolicy:
      nodeSchedulerPolicy: binpack
      gpuSchedulerPolicy: binpack
  ```

- Optionally dedicate certain nodes to whole-GPU workloads using node labels/taints
  (e.g. reserve high-memory nodes for training, use others for sliced inference).

### 7.6 Tuning the split count

The number of vGPUs per physical card is controlled by `deviceSplitCount`
(default 10). Raising it allows more small pods per card; lowering it reduces
overcommit. Adjust based on the typical memory footprint of the workloads so that the
split count and per-pod memory limits reach capacity at roughly the same point.

### 7.7 Operational notes & caveats

- **One GPU plugin per node.** Never run the HAMi device plugin and the official
  NVIDIA device plugin on the same node.
- **Multi-GPU requests need distinct physical cards.** `nvidia.com/gpu: N` requires N
  free physical GPUs on one node; it will not combine slices of a single card.
- **glibc caveat.** The in-container isolation library depends on a private glibc
  symbol removed in glibc 2.34. This affects the **workload container image**, not the
  host OS. Very new base images may need attention; the host distribution version is
  irrelevant to this issue.
- **MIG.** On MIG-capable cards, hardware partitioning is supported in limited modes;
  software slicing (the approach above) is the simpler, uniform default and can be
  used across MIG and non-MIG cards alike.
- **Control-plane nodes running workloads.** If a control-plane node also serves GPU
  workloads, watch its CPU/memory so GPU pods don't starve control-plane components.

---

## Appendix: Quick Reference

**Resource fields**

| Field | Unit | Meaning |
|-------|------|---------|
| `nvidia.com/gpu` | count | Number of **physical** GPUs the pod needs |
| `nvidia.com/gpumem` | MiB | Memory slice **per physical GPU** (1 GB = 1024 MiB) |
| `nvidia.com/gpumem-percentage` | % | Memory slice as a percentage of each card |
| `nvidia.com/gpucores` | % | Compute cap (percentage of the card) |

**Common commands**

```bash
# HAMi pods
kubectl -n kube-system get pods | grep hami

# node vGPU capacity
kubectl get node <gpu-node> -o jsonpath='{.status.capacity.nvidia\.com/gpu}{"\n"}'

# node GPU inventory
kubectl get node <gpu-node> -o yaml | grep -i hami.io/node-nvidia-register

# per-pod physical GPU mapping
kubectl get pods -l <selector> -o json \
  | jq -r '.items[] | .metadata.name + " -> " + (.metadata.annotations["hami.io/vgpu-devices-allocated"] // "none")'

# in-container verification
kubectl exec -it <pod> -- nvidia-smi --query-gpu=memory.total --format=csv
```

**Reference:** HAMi is a CNCF project. Official documentation:
`https://project-hami.github.io/HAMi/` and `https://project-hami.io/docs/`.

---

*Placeholders such as `<gpu-node>`, `<pod>`, `<selector>`, `GPU-xxxx`, and
`<vX.YY.Z>` should be replaced with values specific to your environment.*
