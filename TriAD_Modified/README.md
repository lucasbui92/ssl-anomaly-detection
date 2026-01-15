# Running Instructions for TriAD (Modified Versions)

This repository contains **modified implementations of the TriAD model** used in the accompanying dissertation for smart-grid harmonic anomaly detection. Detailed, pipeline-specific instructions for running **TriAD** should be followed as described in their respective sections.

---

## Acknowledgement

The implementations provided here using the original **TriAD** codebases have been **modified and extended** for smart-grid harmonic anomaly detection experiments.

---

## Environment, Dependencies, and GPU Requirement

> ⚠️ **Important – Read Before Running**
>
> * **TriAD utilizes GPU acceleration via CUDA by default**.
> * Users must ensure that a **CUDA-compatible GPU**, appropriate **GPU drivers**, and a **CUDA-enabled deep learning framework** (e.g., PyTorch with CUDA support) are correctly installed.
> * Failure to configure CUDA properly may result in runtime errors or crashes.
>
> **Dependency Notice**
>
> * All experiments should be executed inside a **Python virtual environment**.
> * Users are responsible for installing all required **Python libraries, frameworks, and system dependencies**.
> * Missing or incompatible dependencies may lead to execution failures.

---

# Running TriAD (Modified Version)

## 1. Navigate to the TriAD Directory

At the terminal, change into the `TriAD_Modified` folder:

```bash
cd TriAD_Modified
```

---

## 2. Create and Activate Virtual Environment

Create a Python virtual environment (if not already created):

```bash
python -m venv venv
```

Activate the environment:

* **Linux / macOS:**

  ```bash
  source venv/bin/activate
  ```
* **Windows (PowerShell):**

  ```bash
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (CMD):**

  ```bash
  venv\Scripts\activate
  ```

---

## Pre-run Preparation

### 3. Prepare Metrics File (Before First Run)

Navigate to:

```
merlin_res/all_metrics.csv
```

* Delete **all rows** starting with:

  ```
  any,Voltage
  ```
* **Do NOT delete the header row**.

This step prevents duplicated or stale evaluation results.

---

## Running the TriAD Pipeline

### 4. Run Feature Selection and Training

From the root of `TriAD_Modified`, execute:

```bash
py feature_selection.py
```

Wait for the process to complete fully before proceeding.

---

## Cleanup and Re-running

If you intend to **rerun the pipeline**, complete **both steps below before re-execution**.

### 5. Delete Generated Evaluation Images

Navigate to:

```
eval_demo/
```

* Delete **all files ending with** `.png`

> This step ensures that system memory is not overburdened by accumulated image files.

---

### 6. Reset Metrics File

Reopen:

```
merlin_res/all_metrics.csv
```

* Delete all rows starting with:

  ```
  any,Voltage
  ```
* Keep the header row intact.

Failure to reset this file may lead to incorrect or duplicated evaluation results.

---

## Modifying Configuration Parameters

Before running experiments with different settings, adjust the configuration files located in the `configs` folder.

### 7. Training Configuration

Open:

```
configs/triad_train_smartgrid_major
```

You may modify parameters such as:

* `cycles`
* `stride_ratio`
* `alpha`

These parameters control training dynamics and windowing behavior.

---

### 8. Dataset and Mode Settings

Open:

```
configs/grid_settings.py
```

**Parameter descriptions:**

* **`LABEL`**: Specifies the label mode being targeted.
* **`TARGET`**: Defines the *maximum number of features* allowed to be combined in a single set.
* **`MULTIVARIATE`**:

  * `False` → Univariate setting (single signal/channel)
  * `True` → Multivariate setting (multiple signals/channels processed jointly)

Changes to configuration files take effect only after re-running the pipeline.

---

## Notes

* Always ensure the virtual environment is activated before running TriAD scripts.
* CUDA/GPU support is required by default.
* Clean `eval_demo` and reset `all_metrics.csv` before rerunning to ensure reproducibility.

---

For further details, refer to the code comments.
