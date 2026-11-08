# Titanic

A notebook for the Kaggle Titanic survival problem: predict whether a passenger survived from class, sex, age, family size, fare and port of embarkation.

## Data

`Titanic-Dataset.csv` has the 891 labelled passengers (12 columns, `Survived` is the target). Age has 177 missing values and Cabin is mostly empty.

## What the notebook does

1. Reads the training and test CSVs, checks missing values and drops `Cabin`, `Name` and `Ticket`.
2. Plots survival against age, sex and embarkation port. In the training data 233 of 314 women survived against 109 of 577 men.
3. Fills missing numeric values with the median and missing `Embarked` with `S`, and encodes the text columns as integers.
4. Fits Linear Regression, Logistic Regression and Random Forest on the whole training set, predicts the Kaggle test set and thresholds the outputs at 0.5.

There is no score in the notebook: the Kaggle test set has no labels and the training data is not split for validation. Adding a validation split is the obvious next step.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Titanic.ipynb
```

The notebook reads `Titanic/train.csv` and `Titanic/test.csv`. The repo has `Titanic-Dataset.csv` (the training data) but not the test file, so download `test.csv` from Kaggle and change the paths to match your folders.
