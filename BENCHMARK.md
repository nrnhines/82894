# nrntraub GPU benchmark (ModelDB 82894)

1/10 network (356 cells), 100 ms, no gap junctions. Compares NEURON CPU,
NEURON native GPU, CoreNEURON CPU, and CoreNEURON GPU spike rasters and runtime.

## Build mechanisms

Requires a GPU-capable NEURON install (`NRN_ENABLE_GPU`) and NVHPC on the PATH.

```bash
cd ~/models/82894
unset N PYTHONPATH
nrnenv core-neuron-gpu gpu-grok   # or your GPU NEURON install
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin:$PATH"
export NMODLHOME="$(readlink -f $N)"
export NMODL_PYLIB="$(find_libpython)"
rm -rf x86_64
nrnivmodl -coreneuron mod
```

Produces `./x86_64/special` (NEURON + native GPU) and `./x86_64/special-core`.

## Run

```bash
python3 run_benchmark.py
```

Results: `results/<case>/`, summary table in `results/summary.txt`.

Override site paths in `config.yaml` (`nrn_bin`, `NVHPC_BIN`). Run a subset:

```bash
python3 run_benchmark.py --cases neuron_gpu_native coreneuron_gpu
```

## Cases

| Case | Flags (via `init.hoc`) |
|------|------------------------|
| `neuron_cpu` | `enable_gpu=0` |
| `neuron_gpu_native` | `enable_gpu=1`, `gpu_backend=native` (default in init.hoc) |
| `coreneuron_cpu` | `coreneuron=1`, `coreneuron_gpu=0` |
| `coreneuron_gpu` | `coreneuron=1`, `coreneuron_gpu=1` |

Reference raster: `neuron_cpu` (4474 spikes for this configuration).

Native GPU requires NEURON with auto cell-permute on `gpu_enable` (Phase B adoption
branch); no model-side `optimize_node_order` call is needed.