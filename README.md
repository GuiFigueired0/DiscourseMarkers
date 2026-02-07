# Discourse Markers in LLMs

This repository contains the codebase for research conducted at **UNIMORE** (University of Modena and Reggio Emilia) by **Guilherme Augusto Rocha de Figueiredo**, as part of the **PNRR-TNE project “Green & Pink for Sustainable Education – GPSEducation”**.

## 📖 Research Overview

This research investigates whether **Discourse Markers (DMs)** classification can improve the performance of Large Language Models (LLMs) on downstream tasks. The project implements a pipeline to:

1. **Mine** discourse markers from Wikipedia dumps.
2. **Process** them into structured datasets.
3. **Utilize** them via **Transfer Learning** or **Multi-Task Learning (MTL)** to enhance model performance on various downstream tasks.

## 📂 Repository Structure

The codebase is organized as follows:

* **`mining/`**: Contains scripts to mine discourse markers from Wikipedia and process them into a dataset format (`sentence1`, `sentence2`, `dm`, `label`).
It also includes discourse marker lists for **English**, **Portuguese**, and **Italian**.

* **`src/`**: The main source code folder containing the training pipeline, models, and experiment runners.
* **`data.zip`**: Contains the mined discourse marker datasets for English, Portuguese, and Italian.
* **`experiments.zip`**: (Legacy) Contains initial experimental code and backups from the early stages of the research.

---

## 🚀 Getting Started

### Prerequisites

We recommend using a dedicated Python environment. You can install all necessary dependencies using the `requirements.txt` file.

```
pip install -r requirements.txt
```

### Dataset Setup

The `src/data` folder is intended to store the datasets required for downstream experiments.

> **Note:** Due to size constraints, the actual datasets are not hosted directly on GitHub. Please refer to the **`report.pdf`** for detailed instructions on where to download them. Some of the datasets also went through a filtering phase. You can find the script for filtering the data inside the **`src/data`** folder.

---

## 💻 Usage

All main commands should be executed from within the `src/` folder.

### 1. Training a Discourse Marker Model

To train a base model specifically on discourse marker classification:

```
cd src
python dm.py --language en

```

*Available languages: `en`, `pt`, `it`.*

### 2. Running Downstream Experiments

To run experiments using Multi-Task Learning (MTL) or Transfer Learning on downstream tasks (e.g., NLI, Topic Classification):

```
cd src
python run_experiments.py --task nli_multi --mode mtl --language en

```

**Arguments:**

* `--task`: The downstream task to run, here represented by the name of the dataset (e.g., `nli_multi`, `paraphrase`).
* `--mode`: The training mode:
  * `baseline`: Standard fine-tuning.
  * `transfer`: Transfer learning from the DM model.
  * `mtl`: Multi-task learning with DM classification.
* `--language`: The language of the dataset (`en`, `pt`, `it`).

---

## 📊 Results

The **`src/results`** folder contains tables and plots summarizing the findings of the experiments conducted so far. Detailed analysis and interpretation of these results can be found in the main research report.

## 📄 Documentation

For a complete explanation of the methodology, pipeline architecture, and detailed results, please refer to the **`report.pdf`** included in this GitHub.

---

*This project is part of the PNRR-TNE “Green & Pink for Sustainable Education – GPSEducation” initiative.*