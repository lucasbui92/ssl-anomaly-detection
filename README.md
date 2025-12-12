# Running TriAD (Modified Version)

This repository contains a modified implementation of **TriAD** for anomaly detection. Follow the steps below carefully to ensure correct execution and reproducible results.

---

## 1. Navigate to the Project Directory

Open a terminal and change into the `TriAD_Modified` folder:

```bash
cd TriAD_Modified
```

---

## 2. Set Up and Activate the Virtual Environment

It is **strongly recommended** to run the code inside a virtual environment.

### Activate the virtual environment

* **Windows (PowerShell):**

  ```bash
  .\venv\Scripts\Activate.ps1
  ```

* **Windows (CMD):**

  ```bash
  venv\Scripts\activate
  ```

* **Linux / macOS:**

  ```bash
  source venv/bin/activate
  ```

### Install required dependencies

Once the environment is activated, install all required libraries and frameworks:

```bash
pip install -r requirements.txt
```

Ensure that all dependencies install successfully before proceeding.

---

## 3. Prepare Metrics File (Before First Run)

1. Navigate to the following folder:

   ```
   merlin_res/
   ```
2. Open the file:

   ```
   all_metrics.csv
   ```
3. **Delete all rows that start with:**

   ```
   any,Voltage
   ```
4. **Do NOT delete the header row.**

This step is required to prevent duplicated or stale evaluation results.

---

## 4. Run Feature Selection

From the root of `TriAD_Modified`, execute:

```bash
py feature_selection.py
```

Wait until the process finishes completely before proceeding.

---

## 5. Required Cleanup Before Re-running

If you intend to **run the pipeline again**, perform **both** steps below **before re-execution**.

### 5.1 Delete Generated Evaluation Images

1. Go to the folder:

   ```
   eval_demo/
   ```
2. Delete **all files ending with** `.png`

> This step ensures that system memory is not overburdened by accumulated image files.

### 5.2 Reset Metrics File Again

1. Reopen:

   ```
   merlin_res/all_metrics.csv
   ```
2. Delete all rows starting with:

   ```
   any,Voltage
   ```
3. Keep the header row intact.

---

## 6. Modifying Configuration Parameters

Before running experiments with different settings, adjust the configuration files located in the `configs` folder.

### 6.1 Training Configuration

Open:

```
configs/triad_train_smartgrid_major
```

You may modify parameters such as:

* `cycles`
* `stride_ratio`
* `alpha`

These parameters control the training dynamics and windowing behavior of TriAD.

### 6.2 Grid and Data Settings

Open:

```
configs/grid_settings.py
```

You may change:

* `LABEL`
* `TARGET`
* `MULTIVARIATE`

These settings control dataset labeling, prediction targets, and whether the model operates in multivariate mode.

---

## Notes

* Always ensure the virtual environment is activated before running any script.
* Re-running without cleaning `eval_demo` or `all_metrics.csv` may lead to incorrect results or excessive memory usage.
* Configuration changes take effect **only after re-running** the pipeline.

---

For questions or issues, please refer to the code comments or contact the repository maintainer.
