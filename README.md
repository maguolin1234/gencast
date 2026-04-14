# GenCast on HPC (CPU-only, Offline, Patched Notebook)

A practical SOP for running the **GenCast mini demo** on an **HPC CPU-only environment** using an **offline notebook** plus a **patched notebook workflow**.

This guide follows the route that actually worked:
- install dependencies in a clean conda env
- install `dinosaur` and `graphcast` from local source
- patch the offline CPU notebook
- run it with `nbconvert`
- verify success from the generated `.ran.ipynb` and log

---

## 1. Scope

This SOP is for:

- HPC
- CPU-only
- offline or semi-offline setup
- GenCast mini demo notebook
- batch execution via `jupyter nbconvert`

This SOP is **not** for:
- GPU training
- full ERA5 training pipeline
- production deployment

---

## 2. Recommended directory layout

```bash
~/gencast/
  graphcast-main/
  dinosaur-main/
  notebooks/
```

You may also keep the notebook inside `graphcast-main/`, but use a consistent naming scheme.

---

## 3. Create environment

```bash
conda create -n gencast -y python=3.10 pip
conda activate gencast
unset PYTHONPATH
```

Install base scientific packages:

```bash
conda install -c conda-forge -y \
  numpy scipy pandas xarray dask netcdf4 h5netcdf \
  matplotlib ipywidgets jupyter nbconvert notebook \
  tqdm rich einops \
  pyproj shapely \
  libspatialindex rtree \
  trimesh
```

Validate `rtree` and `trimesh`:

```bash
python -c "import rtree; import trimesh; print('rtree+trimesh OK')"
```

Install JAX CPU and other Python dependencies:

```bash
python -m pip install -U "jax[cpu]" jaxlib
python -m pip install -U dm-haiku flax optax chex
python -m pip install -U jraph
python -m pip install -U absl-py ml-collections dm-tree
```

Optional:

```bash
python -m pip install -U google-cloud-storage
```

Run a quick environment self-check:

```bash
python - <<'PY'
import jax,jaxlib,haiku,flax,optax,chex
import numpy,scipy,xarray,dask,matplotlib,ipywidgets
import trimesh,rtree,jraph
print("BASIC_IMPORTS_OK")
PY
```

---

## 4. Get source code

### Option A: HPC can access GitHub

```bash
mkdir -p ~/gencast
cd ~/gencast
git clone https://github.com/google-deepmind/graphcast.git graphcast-main
git clone https://github.com/google-research/dinosaur.git dinosaur-main
```

### Option B: HPC cannot access GitHub

On a machine with internet access:

```bash
git clone --depth 1 https://github.com/google-deepmind/graphcast.git graphcast-main
git clone --depth 1 https://github.com/google-research/dinosaur.git dinosaur-main
tar czf graphcast-main.tgz graphcast-main
tar czf dinosaur-main.tgz dinosaur-main
```

Upload both tarballs to the HPC, then extract:

```bash
mkdir -p ~/gencast
cd ~/gencast
tar xzf graphcast-main.tgz
tar xzf dinosaur-main.tgz
```

---

## 5. Install local source packages

Install `dinosaur` first:

```bash
cd ~/gencast/dinosaur-main
python -m pip install -e .
python -c "import dinosaur; from dinosaur import spherical_harmonic; print('dinosaur OK')"
```

Then install `graphcast`:

```bash
cd ~/gencast/graphcast-main
python -m pip install -e .
```

Validate core imports:

```bash
python - <<'PY'
from graphcast import gencast, denoiser, rollout
print("GRAPHCAST_IMPORT_OK")
PY
```

Full import check:

```bash
python - <<'PY'
import trimesh,rtree,jraph,dinosaur
from graphcast import gencast, denoiser, rollout
print("ALL_IMPORTS_OK")
PY
```

---

## 6. Notebook naming convention

Recommended names:

- original: `gencast_mini_demo_cpu.local.ipynb`
- offline CPU: `gencast_mini_demo_cpu.offline.cpu.ipynb`
- patched: `gencast_mini_demo_cpu.offline.cpu_patched.ipynb`
- executed result: `gencast_mini_demo_cpu.offline.cpu_patched.ran.ipynb`
- log: `run_gencast_cpu.offline.cpu_patched.log`

---

## 7. Why the original offline CPU notebook fails

The original CPU offline notebook may fail during the autoregressive rollout with:

```text
ValueError: Only interpret mode is supported on CPU backend.
```

The root cause is that the rollout still reaches a splash-attention / pallas path that is not suitable for normal CPU lowering.

---

## 8. Required notebook patches

### Patch 1: force attention to `mha`

Do **not** rely on splash attention on CPU.

Inside the notebook, make sure the attention path is set to standard multi-head attention:

```python
random_attention_type = _W("mha")
```

and/or hard-code the config field as:

```python
attention_type="mha"
```

Do **not** use:
- `splash`
- `splash_mha`
- `flash`

Also remove any incorrect attempt like:

```python
pltpu.force_tpu_interpret_mode()
```

That is not the correct fix for this CPU route.

### Patch 2: rollout wrapper must accept `forcings`

The rollout code calls the predictor with keyword arguments like:

```python
predictor_fn(rng=..., inputs=..., targets_template=..., forcings=...)
```

So the wrapper must accept `forcings` explicitly.

Use:

```python
def predictor_fn(*, rng, inputs, targets_template, forcings):
    return run_forward_jitted(rng, inputs, targets_template, forcings)
```

Then pass this wrapper into the rollout generator.

### Patch 3: if `num_ensemble_members=1`, skip CRPS

If you reduce the ensemble size to 1 for a quick CPU smoke test, CRPS will fail because it needs at least 2 samples.

Use one of these approaches:

#### Option A: skip CRPS when sample size < 2

```python
if predictions.sizes.get("sample", 1) < 2:
    print("sample<2, skip CRPS; set num_ensemble_members>=2 to enable CRPS.")
    data = {
        "Targets": ...,
        "Ensemble Mean": ...,
    }
else:
    data = {
        "Targets": ...,
        "Ensemble Mean": ...,
        "Ensemble CRPS": ...,
    }
```

#### Option B: set `num_ensemble_members` to 2 or 4

For CPU-first testing, start with:

```python
num_ensemble_members = 1
num_steps_per_chunk = 1
```

Once the workflow runs end-to-end, increase the ensemble size.

---

## 9. Run the patched notebook

Optional: clear old outputs first.

```bash
jupyter nbconvert --clear-output --inplace gencast_mini_demo_cpu.offline.cpu_patched.ipynb
```

Run the notebook in batch mode:

```bash
JAX_PALLAS_INTERPRET=1 \
jupyter nbconvert --to notebook --execute gencast_mini_demo_cpu.offline.cpu_patched.ipynb \
  --ExecutePreprocessor.kernel_name=python3 \
  --output gencast_mini_demo_cpu.offline.cpu_patched.ran.ipynb \
  --ExecutePreprocessor.timeout=7200 \
  --log-level=INFO 2>&1 | tee run_gencast_cpu.offline.cpu_patched.log
```

If the notebook is in another directory, use an absolute path.

---

## 10. Minimal pre-run checklist

Before running:

```bash
conda activate gencast
which python
python -c "import trimesh,rtree,jraph,dinosaur; from graphcast import gencast"
ls gencast_mini_demo_cpu.offline.cpu_patched.ipynb
```

Also confirm:
- the notebook uses `attention_type="mha"` or `random_attention_type = _W("mha")`
- the rollout wrapper accepts `forcings`
- CRPS is guarded if `num_ensemble_members=1`

---

## 11. Success criteria

The run is considered successful when both conditions are met:

```bash
grep -nE "CellExecutionError|Traceback|ValueError" run_gencast_cpu.offline.cpu_patched.log || echo "NO_ERRORS"
```

and

```bash
ls -lh gencast_mini_demo_cpu.offline.cpu_patched.ran.ipynb
```

If `NO_ERRORS` is printed and the `.ran.ipynb` exists, the patched CPU offline workflow has completed successfully.

---

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'jraph'`

```bash
python -m pip install -U jraph
```

### `ModuleNotFoundError: No module named 'dinosaur'`

Install from local source:

```bash
cd ~/gencast/dinosaur-main
python -m pip install -e .
```

### `ModuleNotFoundError: No module named 'rtree'`

```bash
conda install -c conda-forge -y libspatialindex rtree
python -c "import rtree; print('rtree OK')"
```

If conda says it is installed but Python still cannot import it, verify:

```bash
which python
conda list | grep -E 'rtree|libspatialindex'
```

### `ValueError: Only interpret mode is supported on CPU backend`

This means the notebook is still reaching a splash attention / pallas path.

Fix:
- make sure attention is `mha`
- do not rely on splash attention
- optionally keep `JAX_PALLAS_INTERPRET=1` in the execution command

### `TypeError: <lambda>() got an unexpected keyword argument 'forcings'`

Your rollout wrapper does not accept the keyword argument `forcings`.

Use:

```python
def predictor_fn(*, rng, inputs, targets_template, forcings):
    return run_forward_jitted(rng, inputs, targets_template, forcings)
```

### `ValueError: predictions must have dim 'sample' with size at least 2`

You are trying to compute CRPS with only one ensemble member.

Fix:
- either skip CRPS when sample size < 2
- or increase `num_ensemble_members` to at least 2

### `IndentationError: unexpected indent`

Clean up the notebook cell indentation. This usually happens after pasting code with an extra tab or extra leading spaces.

---

## 13. One-command install script (online HPC)

Save as `install_gencast_cpu.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

conda create -n gencast -y python=3.10 pip
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gencast
unset PYTHONPATH

conda install -c conda-forge -y \
  numpy scipy pandas xarray dask netcdf4 h5netcdf \
  matplotlib ipywidgets jupyter nbconvert notebook \
  tqdm rich einops \
  pyproj shapely \
  libspatialindex rtree \
  trimesh

python -m pip install -U "jax[cpu]" jaxlib
python -m pip install -U dm-haiku flax optax chex jraph absl-py ml-collections dm-tree

mkdir -p ~/gencast && cd ~/gencast
git clone https://github.com/google-deepmind/graphcast.git graphcast-main
git clone https://github.com/google-research/dinosaur.git dinosaur-main

cd ~/gencast/dinosaur-main
python -m pip install -e .

cd ~/gencast/graphcast-main
python -m pip install -e .

python - <<'PY'
import jax,jaxlib,haiku,flax,optax,chex
import numpy,scipy,xarray,dask,matplotlib,ipywidgets
import trimesh,rtree,jraph
import dinosaur; from dinosaur import spherical_harmonic
from graphcast import gencast, denoiser, rollout
print("ALL_IMPORTS_OK")
PY
```

Run with:

```bash
bash install_gencast_cpu.sh
```

---

## 14. Final notes

This repository documents the **working CPU-only HPC route** for the GenCast mini demo:
- local source install
- offline notebook
- patched notebook
- batch execution via `nbconvert`

If you later move to GPU, you should revisit the attention path and performance settings instead of assuming this CPU-safe patch is the best long-term configuration.

---

## 15. Final working route on the HPC: where it ran, how it was submitted, and how to inspect results

The final successful route was **not** the original offline CPU notebook and **not** an MPI-style shell script such as `run_test1.sh`.

The actual successful route was:

- working directory: `/data/home/maguolin/gencast/graphcast-main`
- notebook: `gencast_mini_demo_cpu.offline.cpu_patched.ipynb`
- execution method: `jupyter nbconvert --execute ...`
- main outputs:
  - `gencast_mini_demo_cpu.offline.cpu_patched.ran.ipynb`
  - `gencast_mini_demo_cpu.offline.cpu_patched.ran.html`
  - `run_gencast_cpu.offline.cpu_patched.log`

### 15.1 Where to run

Use the `graphcast-main` directory as the working directory:

```bash
cd /data/home/maguolin/gencast/graphcast-main
```

Environment preparation, notebook editing, and file checking can be done on the login node.
For the actual notebook execution, the recommended route is to submit the job to a Slurm compute node.

### 15.2 How to submit

The core execution command is:

```bash
JAX_PALLAS_INTERPRET=1 \
jupyter nbconvert --to notebook --execute gencast_mini_demo_cpu.offline.cpu_patched.ipynb \
  --ExecutePreprocessor.kernel_name=python3 \
  --output gencast_mini_demo_cpu.offline.cpu_patched.ran.ipynb \
  --ExecutePreprocessor.timeout=14400 \
  --log-level=INFO 2>&1 | tee run_gencast_cpu.offline.cpu_patched.log
```

The recommended HPC submission pattern is to place that command inside a Slurm job script such as:

```bash
run_gencast_cpu48.sbatch
```

and submit it with:

```bash
sbatch run_gencast_cpu48.sbatch
```

### 15.3 Important clarification

The uploaded `run_test1.sh` is **not** the GenCast submission script.
It is an OpenFOAM / terrain solver MPI script and should not be used as the GenCast execution template.

### 15.4 How to monitor the job

If submitted through Slurm, check job status with:

```bash
squeue -u $USER
```

Watch scheduler stdout/stderr with:

```bash
tail -f slurm-<jobid>.out
tail -f slurm-<jobid>.err
```

### 15.5 How to inspect results

The most important result file is the executed notebook:

```bash
gencast_mini_demo_cpu.offline.cpu_patched.ran.ipynb
```

This file contains the executed cells, printed diagnostics, and figures.

For easier viewing outside Jupyter, also open:

```bash
gencast_mini_demo_cpu.offline.cpu_patched.ran.html
```

This is the browser-friendly export of the executed notebook.

The main runtime log is:

```bash
run_gencast_cpu.offline.cpu_patched.log
```

### 15.6 Fast success check

Use the following commands:

```bash
grep -nE "CellExecutionError|Traceback|ValueError" run_gencast_cpu.offline.cpu_patched.log || echo "NO_ERRORS"
ls -lh gencast_mini_demo_cpu.offline.cpu_patched.ran.ipynb
```

Interpretation:

- `NO_ERRORS` means no major execution error was found in the log.
- If the `.ran.ipynb` file exists, the patched CPU offline workflow completed successfully.

### 15.7 One-sentence summary

The final working GenCast route on the HPC was:
run the **patched offline CPU notebook** from `/data/home/maguolin/gencast/graphcast-main` via `jupyter nbconvert --execute ...`, preferably through a Slurm submission script such as `run_gencast_cpu48.sbatch`, then inspect `gencast_mini_demo_cpu.offline.cpu_patched.ran.ipynb`, its HTML export, and the execution log.
