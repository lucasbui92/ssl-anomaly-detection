# Running Instructions for TriAD and CARLA (Modified Versions)

This repository contains **modified implementations of the TriAD and CARLA models** used in the accompanying dissertation for smart-grid harmonic anomaly detection. Detailed, pipeline-specific instructions for running **TriAD** and **CARLA** should be followed as described in their respective sections.

---

## Acknowledgement

The implementations provided here using the original **TriAD** and **CARLA** codebases have been **modified and extended** for smart-grid harmonic anomaly detection experiments.

---

## Environment, Dependencies, and GPU Requirement

> ⚠️ **Important – Read Before Running**
>
> * Both **TriAD and CARLA utilize GPU acceleration via CUDA by default**.
> * Users must ensure that a **CUDA-compatible GPU**, appropriate **GPU drivers**, and a **CUDA-enabled deep learning framework** (e.g., PyTorch with CUDA support) are correctly installed.
> * Failure to configure CUDA properly may result in runtime errors or crashes.
>
> **Dependency Notice**
>
> * All experiments should be executed inside a **Python virtual environment**.
> * Users are responsible for installing all required **Python libraries, frameworks, and system dependencies**.
> * Missing or incompatible dependencies may lead to execution failures.

---

# 1. Running CARLA (Modified Version)

## Environment Setup

### 1. Navigate to the CARLA Directory

At the terminal, change into the `CARLA_Modified` folder:

```bash
cd CARLA_Modified
```

---

### 2. Create and Activate Virtual Environment

Create a Python virtual environment named `carla-env`:

```bash
python -m venv carla-env
```

Activate the environment:

* **Linux / macOS:**

  ```bash
  source carla-env/bin/activate
  ```
* **Windows:**

  ```bash
  carla-env\Scripts\activate
  ```

---

## Pre-run Checks and Setup

### 3. Check Results Folder

Navigate to the `results` folder:

* If the folder is **empty**, no action is required.
* If the folder contains any **nested subfolders**, delete them **before running** the pipeline.

This ensures that stale outputs do not interfere with new experiment results.

---

### 4. Update Dataset Path Configuration

Open the following file:

```
utils/mypath.py
```

#### (a) Add import at the top of the file

```python
from pathlib import Path
```

#### (b) Update the Smart Grid dataset path

Locate the conditional block:

```python
elif database == 'smart_grid':
```

Modify it to include the following:

```python
BASE_DIR = Path(__file__).resolve().parent.parent
return BASE_DIR / "datasets" / "SmartGrid"
```

This ensures correct dataset path for the smart-grid experiments.

---

## Running the CARLA Pipeline

> ⚠️ **Important**: The following commands **must be executed in order**.

### 5. Run Pretext Training

Execute the pretext task:

```bash
py carla_pretext.py \
  --config_env configs/env.yml \
  --config_exp configs/pretext/carla_pretext_smartgrid_major.yml \
  --fname smart_grid_major
```

Wait for the process to complete before proceeding.

---

### 6. Run Classification

After pretext training finishes, execute:

```bash
py carla_classification.py \
  --config_env configs/env.yml \
  --config_exp configs/classification/carla_classification_smartgrid_major.yml \
  --fname smart_grid_major
```

Wait for the run to finish completely.

> ⚠️ These two commands **must be executed sequentially**. Running classification without completing pretext training will result in errors.

---

## Evaluation and Re-running

* After the **classification run**, scroll up in the terminal output to view the **evaluation results**.

### Before Re-running Any Command

If you intend to rerun **either** the pretext or classification stage:

1. Navigate to the `results` folder
2. **Delete all subfolders** inside `results`

Failure to clean the `results` directory may lead to incorrect evaluations or execution errors.

---

## Modifying Configuration Parameters

Before running experiments with different settings, adjust the configuration files located in the `configs` folder.

### 7. Dataset and Window Configuration

Open:

```
configs/smartgrid.yml
```

You may modify:

* `label_mode`
* `feature_columns`: Comment out features you do not need for training
* Sliding window parameters: `size` and `stride`
* Thresholding behavior: `mode` and `value`

---

### 8. Pretext and Classification Hyperparameters

* Open:

  ```
  configs/pretext/carla_pretext_smartgrid_major.yml
  ```

  to modify pretext-task hyperparameters.

* Open:

  ```
  configs/classification/carla_classification_smartgrid_major.yml
  ```

  to modify classification-stage hyperparameters.

Changes to these files take effect only after re-running the pipeline.

---

## Notes

* Always ensure the virtual environment is activated before running any CARLA scripts.
* CUDA/GPU support is required by default.
* Clean the `results` directory before re-running experiments to ensure reproducibility.

---

# 2. Running TriAD (Modified Version)

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

For further details, refer to the code comments or the accompanying dissertation.
