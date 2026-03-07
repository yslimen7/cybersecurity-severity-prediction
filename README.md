# Global Cybersecurity Severity Prediction

## Authors
- Gabriel Bastianello  
- Beatriz Evelbauer  
- Yasmine Slimen  

This challenge was created as part of the DataCamp group project.

---

# Context

Cybersecurity incidents can have very different levels of impact depending on the type of attack, the number of affected users, and the time required to resolve the incident.

The goal of this challenge is to **predict the severity level of a cybersecurity incident** based on a set of characteristics describing simulated attacks.

Severity levels are divided into four classes:

- Low
- Medium
- High
- Critical

The task is therefore a **multi-class classification problem**.

---

# Origin of the Data

The dataset used in this challenge was generated for educational purposes to simulate cybersecurity incidents occurring across different countries and industries.

The dataset creation and preparation are handled by the script: tools/setup_data.py


This script:
- loads the base simulated data
- creates the **severity label**
- splits the dataset into **training**, **public test**, and **private test** sets

The severity label is derived mainly from two impact indicators:

- **Number of Affected Users**
- **Incident Resolution Time (in Hours)**

Higher values of these indicators correspond to higher severity levels.

---

# Data Description

The dataset contains several features describing each cybersecurity incident:

- `Country`
- `Year`
- `Attack Type`
- `Target Industry`
- `Number of Affected Users`
- `Attack Source`
- `Security Vulnerability Type`
- `Defense Mechanism Used`
- `Incident Resolution Time (in Hours)`

These variables include both **categorical** and **numerical** data.

---

# Dataset Structure

The data is organized in a development phase (`dev_phase`) with:

### Training data
`dev_phase/input_data/train/`

- `train_features.csv`
- `train_labels.csv`

### Public test set
`dev_phase/input_data/test/`

- `test_features.csv`

Used for the **public leaderboard**.

### Private test set
`dev_phase/input_data/private_test/`

- `private_test_features.csv`

Used for the **final evaluation**.

### Reference labels

`dev_phase/reference_data/`

- `test_labels.csv`
- `private_test_labels.csv`

These labels are used internally by the scoring program.

---

# Exploratory Data Analysis

The `starting_kit.ipynb` notebook includes an exploratory data analysis (EDA) step to better understand the dataset before training a model.

The analysis consists of:

- inspecting dataset dimensions
- checking column types
- identifying categorical and numerical variables
- verifying the presence of missing values
- analyzing the distribution of the target variable (`label`)

The training dataset contains **2386 samples and 9 input features** describing simulated cybersecurity incidents.

The dataset includes both **categorical features** (such as `Country`, `Attack Type`, `Target Industry`, `Attack Source`, `Security Vulnerability Type`, and `Defense Mechanism Used`) and **numerical features** (`Year`, `Number of Affected Users`, and `Incident Resolution Time (in Hours)`).

The exploratory analysis shows that **no missing values are present in the dataset**, which simplifies preprocessing.

The distribution of the severity labels is also **almost perfectly balanced across the four classes**:

- Low
- Medium
- High
- Critical

This balanced distribution makes the dataset well suited for a multi-class classification problem and ensures that the evaluation metric is not biased toward a dominant class.

To further understand the relationship between the features and the severity level, the notebook visualizes the distribution of two key impact-related variables:

- **Number of Affected Users**
- **Incident Resolution Time (in Hours)**

The boxplots show that incidents with higher severity levels generally affect more users and require longer resolution times, while lower severity incidents tend to have smaller impact and shorter recovery times.

These observations confirm that the dataset contains meaningful patterns linking incident characteristics to the severity label, making the prediction task suitable for machine learning approaches.

---

# Task

Participants must train a machine learning model that predicts the column: 
                label
with one of the following classes:

- Low
- Medium
- High
- Critical

Predictions must be produced for:

- `test`
- `private_test`

---

# Evaluation Metric

Participants are evaluated using the **Macro F1-score**.

The F1-score combines precision and recall:
F1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)

The **Macro F1-score** averages the F1-score across all classes:
Macro F1 = (1 / K) * sum(F1_k)


This metric ensures that **all classes are equally important**, even if the dataset is imbalanced.

Evaluation is performed on:

- `test` for the public leaderboard
- `private_test` for the final ranking

---

# Expected Output

The submission must generate two CSV files:
    test_predictions.csv
    private_test_predictions.csv


Each file must contain a single column: label

with values among:
  Low
  Medium
  High
  Critical


---

# Starting Kit

The notebook `starting_kit.ipynb` provides a baseline pipeline:

- loading the dataset
- basic exploratory data analysis
- preprocessing categorical and numerical features
- training a baseline model
- generating prediction files

Participants can modify and improve this baseline.

---

# Run Locally

### 1 - Ingestion

Generate predictions from the submission.
python ingestion_program/ingestion.py
--data-dir dev_phase/input_data/
--output-dir ingestion_res/
--submission-dir solution/

### 2 - Scoring

Compute the Macro F1-score.
python scoring_program/scoring.py
--reference-dir dev_phase/reference_data/
--prediction-dir ingestion_res/
--output-dir scoring_res/

---

# Submission

A valid submission must:

- train a model using the training data
- predict the `label` column
- generate the files:
  - `test_predictions.csv`
  - `private_test_predictions.csv`

An example implementation is provided in the `solution/` folder.