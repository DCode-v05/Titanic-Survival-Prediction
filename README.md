# Titanic — Machine Learning from Disaster

**A start-to-finish survival classification walkthrough on the Kaggle Titanic dataset: EDA, cleaning, encoding, and three models compared in one notebook.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=white)

## Overview

This is my take on the classic Kaggle problem "Titanic: Machine Learning from Disaster": predict which passengers survived the 1912 sinking from features like age, sex and ticket class. It's a single Jupyter notebook that goes through the whole pipeline: load the data, look at it, clean it, turn the text columns into numbers, train a few models, and turn their outputs into survival predictions.

I built it as a learning project to get comfortable with the standard tabular ML workflow in scikit-learn from start to finish, not to chase a leaderboard score. The dataset has 891 labelled passengers and 12 columns, and the notebook trains three different models on it (Linear Regression, Logistic Regression and a Random Forest classifier) so I could compare how each handles the same problem. It doesn't report a score for any of them, since the training data is never split for validation.

## Key Features

- Loads and inspects the Titanic passenger data (`.info()`, `.describe()`, null counts, row count).
- Exploratory data analysis with pandas crosstabs and Matplotlib/Seaborn plots for Age, Sex and Embarked against survival.
- Missing-value handling: numeric columns filled with the median, the `Embarked` column filled with its most common port (`S`).
- Drops columns that don't help a plain tabular model: `Cabin` (mostly empty), plus `Name` and `Ticket` (free text, not categorical here).
- Manual label encoding: each text column is mapped to integers with a per-column dictionary loop instead of an off-the-shelf encoder.
- Trains three models on the same features: Linear Regression, Logistic Regression and a Random Forest classifier.
- Converts each model's output to a binary survived/not-survived label by thresholding at 0.5, so even the regression output becomes a 0/1 label.
- Collects predictions into per-model `PassengerId → Survived` DataFrames, Kaggle-submission style.

## How It Works

The whole thing lives in `Titanic.ipynb` (55 cells). It follows a straight data-to-prediction pipeline.

### 1. Load and explore

The notebook reads the training data, then checks its shape and health with `len(df)`, `.info()`, `.describe()` and a per-column `isna().sum()` to find the missing values. The three columns with real gaps are `Age` (177 missing), `Cabin` and `Embarked`.

EDA is done mostly with `pd.crosstab` plus bar and count plots:

- **Sex vs Survived:** the strongest signal in the data.
- **Embarked vs Survived:** survival counts split by port of embarkation (C / Q / S).
- **Age vs Survived:** a crosstab across 88 distinct age values, plus a count plot.

### 2. Clean the data

- `Cabin` is dropped because most of its values are missing.
- `Name` and `Ticket` are dropped. They're high-cardinality free text, not useful categories for this model.
- Numeric columns with gaps (e.g. `Age`) are filled with the column median, via a loop that checks each column's dtype.
- `Embarked` is filled with `"S"`, the most frequent port.

### 3. Encode the categoricals

Instead of `LabelEncoder` or one-hot, the notebook does its own encoding: for every object-dtype column, it walks the unique values, builds a `{value: integer}` dictionary, and replaces the strings in place. This is applied to the train and test frames separately.

### 4. Train the models

Features are split into `X` (everything except `Survived`) and `y` (the `Survived` column). Three models are fit on the same `X`/`y`:

1. `LinearRegression`
2. `LogisticRegression`
3. `RandomForestClassifier`

Each one predicts on the Kaggle test frame.

### 5. Turn predictions into labels

Predictions are packed into DataFrames keyed by `PassengerId`. Linear Regression returns continuous values, and to keep all three consistent, every prediction is thresholded with `np.where(pred >= 0.5, 1, 0)` to give a final 0/1 survival label per passenger.

## Results / Highlights

This is a training/inference notebook, not an evaluation harness. The Kaggle test set has no ground-truth labels and the training data isn't split for validation, so the notebook computes no accuracy or classification report. The metric imports are there but unused. Adding a validation split is the obvious next step. What the notebook does produce is the EDA, where the survival patterns show up in the crosstab counts:

- **Sex is the dominant factor.** Of 314 women, 233 survived and 81 did not (~74% survived). Of 577 men, only 109 survived against 468 lost (~19% survived).
- **Port of embarkation tracks survival too.** Cherbourg (C): 93 survived / 75 lost. Queenstown (Q): 30 / 47. Southampton (S): 217 / 427.
- **Children skew toward surviving.** The youngest age bins in the Age-vs-Survived crosstab (passengers under ~1 year) survived almost across the board.
- **Scale:** 891 passengers, 12 original columns, reduced to a numeric feature set after dropping `Cabin`, `Name` and `Ticket`; 3 models trained for comparison.

## Tech Stack

- **Language:** Python 3.x
- **Notebook:** Jupyter
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML:** scikit-learn (`LinearRegression`, `LogisticRegression`, `RandomForestClassifier`)

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook (via Anaconda or pip)

### Installation

```bash
git clone https://github.com/DCode-v05/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running

```bash
jupyter notebook Titanic.ipynb
```

> Note on data paths: the notebook reads `Titanic/train.csv` and `Titanic/test.csv`, the two-file Kaggle download. This repo only has `Titanic-Dataset.csv` (the 891 labelled training rows, same columns as `train.csv`) and no test file. To run the notebook, download `test.csv` from Kaggle and change the `read_csv` paths to match your folders, pointing the training one at `Titanic-Dataset.csv` if you like.

## Usage

Open `Titanic.ipynb` and run the cells top to bottom. The flow is:

1. **Load** the data and print its info, description and null counts.
2. **Explore** survival against Age, Sex and Embarked with crosstabs and plots.
3. **Clean:** drop `Cabin`/`Name`/`Ticket`, fill numeric nulls with the median and `Embarked` with `S`.
4. **Encode** the remaining text columns to integers.
5. **Train** the three models on `X` (features) and `y` (`Survived`).
6. **Predict** on the test data and read off the per-passenger survived/not-survived labels.

To experiment, swap in different features, try other scikit-learn models, or add a labelled validation split so you can score accuracy/precision/recall (the metric imports are already in the first cell).

## Project Structure

```
Titanic-Survival-Prediction/
├── Titanic.ipynb         # Full workflow: EDA → cleaning → encoding → 3 models → predictions (55 cells)
├── Titanic-Dataset.csv   # Labelled training data, 891 rows × 12 columns (no test file)
├── .gitignore
└── README.md             # Project documentation
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
