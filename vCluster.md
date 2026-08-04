vCluster provides isolated Kubernetes tenant clusters. each team has own kubernetes API.
Like, Virtual clusters.

### Sample setup:
1. Suppose Your host has : 4 x H100 GPUs
2. HAMi Advertises : nvidia.com/gpu=40
3. Now create Three vClusters :
```
vcluster-team-A
vcluster-team-B
vcluster-team-C
host-cluster
```

MiniCube - Creates real kubernetes clusters ( Control Plane + worker node) on single machine.

vCluster - creates virtual kubernetes control plane inside existing kubernetes cluster.
- it does not  create another worker node.
- One control plane will be there on host machine. and vluster creates multiple control plane in pods.