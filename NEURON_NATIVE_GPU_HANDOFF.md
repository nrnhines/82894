# NEURON native GPU handoff — ModelDB 82894 (nrntraub)

Read this document when working in the **Adopting CoreNEURON GPU into NEURON**
worktree (`~/neuron/core-neuron-gpu`, branch `hines-grok/feature/neuron-core-gpu-adoption`,
[NEURON PR #3801](https://github.com/neuronsimulator/nrn/pull/3801)) to diagnose why
**NEURON native GPU** (`gpu.backend="native"`) does not match CPU or CoreNEURON spike
rasters on the nrntraub benchmark model.

Persistent NEURON dev notes: `~/neuron/notes/README.md`, `~/neuron/notes/gpu_workstation.md`.
Phase B contract: `docs/dev/native-gpu-fixed-step.rst` and `docs/dev/native-gpu-adoption/00-overview.md`
in the NEURON tree.

---

## Repository and branch

```bash
git clone git@github.com:nrnhines/82894.git
cd 82894
git fetch origin
git checkout hines-grok/native-gpu
```

Fork of [ModelDBRepository/82894](https://github.com/ModelDBRepository/82894) (Traub
thalamocortical network). Benchmark uses the **1/10 network** (356 cells), no gap junctions,
100 ms, single thread.

---

## NEURON build (GPU-capable install)

On the adoption machine, use the NVHPC GPU build:

```bash
# ~/neuron/core-neuron-gpu, branch hines-grok/feature/neuron-core-gpu-adoption
unset N PYTHONPATH
nrnenv core-neuron-gpu gpu-grok
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin:$PATH"
```

Build tree: `~/neuron/core-neuron-gpu/build-gpu-grok` (`./grok-bld configure|build|test`).

Runtime contract for **native GPU** (Phase B):

```python
from neuron import h, gpu
# coreneuron.enable stays False
gpu.enable = True
gpu.backend = "native"
pc.psolve(tstop)
```

In this model, the same is driven from HOC via `init.hoc` (`enable_gpu=1`, `gpu_backend`
defaults to `"native"`).

---

## Mechanism build: `nrnivmodl -coreneuron`

From the model directory, after activating `nrnenv` and NVHPC:

```bash
cd ~/models/82894   # or your clone path
unset N PYTHONPATH
nrnenv core-neuron-gpu gpu-grok
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin:$PATH"
export NMODLHOME="$(readlink -f $N)"
export NMODL_PYLIB="$(find_libpython)"
rm -rf x86_64
nrnivmodl -coreneuron mod
```

Success produces:

- `x86_64/special` — NEURON (CPU + native GPU)
- `x86_64/special-core` — standalone CoreNEURON
- `x86_64/libnrnmech.so`, `x86_64/libcorenrnmech.so`

Shell `nrnivmodl` does **not** set `NMODLHOME` / `NMODL_PYLIB`; the CMake wrappers
(`nrnivmodl-all-cmake`) do.

### Mod file fixes on `hines-grok/native-gpu` (required for `-coreneuron`)

| File | Issue | Fix |
|------|--------|-----|
| `mod/traub_nmda.mod` | NMODL: `A_`, `BB1_`, `BB2_` in `PARAMETER` but assigned in `INITIAL` | Moved to `ASSIGNED` |
| `mod/ri.mod` | NEURON-only `VERBATIM` (`Section`, `NODEA`, …) fails CoreNEURON compile | Wrapped in `#ifndef NRNBBCORE` (no-op on CoreNEURON; OK when `use_traubexact=0`) |

NEURON-only build still works: `nrnivmodl mod` (without `-coreneuron`).

---

## Benchmark harness

```bash
cd ~/models/82894
unset N PYTHONPATH
nrnenv core-neuron-gpu gpu-grok
export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin:$PATH"
python3 run_benchmark.py
```

- Config: `config.yaml` (four cases; override `nrn_bin` / `NVHPC_BIN` on other clusters).
- Driver: `run_benchmark.py` — runs cases, archives `results/<case>/`, sorts spikes,
  diffs against `reference_case: neuron_cpu`, prints table to `results/summary.txt`.
- Use `./x86_64/special`, not plain `nrniv`, on this GPU build.

### `init.hoc` runtime flags (branch)

| Flag | Default | Purpose |
|------|---------|---------|
| `enable_gpu` | 0 | Native GPU (`pc.gpu_enable`, `gpu_backend`, `gpu_download_flush`) |
| `coreneuron` | 0 | Embed CoreNEURON in `pc.psolve` |
| `coreneuron_gpu` | 0 | With `coreneuron=1`, sets `coreneuron.gpu=True`, `permute=2` |
| `benchmark_quiet` | 0 | Skip verbose post-run stats; still writes spikes and `perf.dat` |
| `mytstop` | 1000 | Simulation time (benchmark uses 100) |
| `one_tenth_ncell` | 1 | 356-cell network |
| `use_gap` | 0 | Gap junctions off |

NEURON `-c` cannot assign HOC strings; `gpu_backend="native"` stays the `init.hoc` default.

### Single-case example (NEURON CPU)

```bash
./x86_64/special -c mytstop=100 -c one_tenth_ncell=1 -c use_gap=0 \
  -c nthread=1 -c enable_gpu=0 -c benchmark_quiet=1 init.hoc
```

Spike output: `out1.dat` (`time_ms<TAB>gid`). Compare with `sortspike` (in `$N/bin`).

---

## Correctness status (as of branch `hines-grok/native-gpu`)

| Case | Spikes | Match `neuron_cpu`? | Notes |
|------|--------|---------------------|--------|
| `neuron_cpu` | 4474 | reference | |
| `coreneuron_cpu` | 4474 | **yes** | |
| `coreneuron_gpu` | 4474 | **yes** | ~5× faster than CPU on T1000 |
| `neuron_gpu_native` | 726 | **no** | **Open problem** |

### First spike raster difference (native GPU vs CPU)

Sorted `(time_ms, gid)` — first mismatch:

- **GPU only:** `t=0.45 ms`, `gid=0`
- **CPU next spike:** `t=1.425 ms`, `gid=156`

CPU has **zero** spikes before 1.0 ms; GPU has **726** spikes before 1.0 ms (many gids at
exactly 0.45 ms). This is not a small timing shift — it looks like spurious early activity
(possibly ectopic / NetStim path; `use_ectopic=1` by default).

**gid 0** is `suppyrRS[0]` (first superficial pyramidal RS cell). Ectopic drive uses `PulseSyn`
on `comp[72]` when `use_ectopic=1`.

---

## Diagnostic workflow (native GPU bug)

Recommended order (see
[ParallelContext.prcellstate](https://www.neuronsimulator.org/en/latest/progref/modelspec/programmatic/network/parcon.html)):

1. **Spike raster** — find first `(time, gid)` difference (above).
2. **`prcellstate` at `t=0`** — confirm initialization matches CPU for that gid.
3. **`prcellstate` after first step** (`t=0.025 ms`) — find earliest state divergence.
4. **Binary search on `mytstop`** — narrow time of first `prcellstate` difference for that gid.
5. Optionally dump the **ectopic `S_NetStim` artificial cell** gid (not in gid 0’s
   `prcellstate` synlist dump).

Local helper (may be uncommitted in clone): `prcellstate_diag.hoc`

```bash
./x86_64/special -c enable_gpu=0 -c prcellstate_gid=0 prcellstate_diag.hoc
./x86_64/special -c enable_gpu=1 -c prcellstate_gid=0 prcellstate_diag.hoc
# outputs: cs0.0.1.init, cs0.0.1.step1 (rename between runs)
```

**Finding so far:** CPU vs GPU `prcellstate` for gid 0 are **identical at `t=0`**; they
diverge after **one** `fadvance` (small numerical diffs in `v` and mechanisms at `t=0.025 ms`).
That alone does not explain the 0.45 ms spike burst — investigate net events / spike recording
on the native GPU path next.

### Isolating ectopic

Re-run benchmark with `-c use_ectopic=0` (add to `config.yaml` or command line) to see if
early GPU spikes disappear.

---

## Performance (T1000, 100 ms, 356 cells — indicative only)

| Case | ~Runtime | vs CPU |
|------|----------|--------|
| NEURON CPU | ~52 s | 1.0× |
| CoreNEURON CPU | ~31 s | 1.6× |
| CoreNEURON GPU | ~9 s | 5.3× |
| NEURON native GPU | ~50 s | ~1.0× (wrong spikes) |

Native GPU on this model is neither fast nor correct yet; CoreNEURON GPU is the working GPU
baseline.

---

## Suggested investigation areas in NEURON (PR #3801)

- Native GPU spike recording / `ParallelNetManager` interaction at psolve start
- `NET_RECEIVE` / `net_send` / ectopic `NetStim` delivery on device vs CPU queues
- `gpu.download_flush_interval` and host visibility of spike buffers
- Compare against passing `*_py_gpu_native` modtests patterns in `docs/dev/gpu-testing.rst`

---

## Git commits on `hines-grok/native-gpu`

1. HOC/mod: `init.hoc` GPU/CoreNEURON flags; `traub_nmda.mod`, `ri.mod` CoreNEURON build fixes.
2. `config.yaml`, `run_benchmark.py` — benchmark driver.
3. This handoff document.

Remote: `git@github.com:nrnhines/82894.git`, branch `hines-grok/native-gpu`.