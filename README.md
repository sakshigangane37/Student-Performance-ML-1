# Student Performance Machine Learning - 

## Project Description

This project performs Machine Learning analysis on student academic and behavioral data using a Decision Tree Classifier.

The objective is to analyze different student-related features and predict whether a student will Pass or Fail.

## Dataset

The dataset used in this project is:

`student_performance_ml.csv`

The dataset contains the following features:

- `StudyHours` - Number of hours a student studies per day.
- `Attendance` - Percentage of class attendance.
- `PreviousScore` - Marks obtained in the previous examination.
- `AssignmentsCompleted` - Number of assignments completed by the student.
- `SleepHours` - Average number of hours a student sleeps per day.
- `FinalResult` - Target variable.
  - `1` = Pass
  - `0` = Fail

## Machine Learning Model

The project uses:

- Decision Tree Classifier
- Train-Test Split
- Accuracy Score
- Feature Importance
- Decision Tree Visualization

## Project Features

This project performs the following analysis:

1. Calculates feature importance using `feature_importances_`.
2. Identifies the most and least important features.
3. Removes `SleepHours` and compares model accuracy.
4. Trains the model using only `StudyHours` and `Attendance`.
5. Predicts the results of 5 new students.
6. Calculates accuracy manually without using `accuracy_score`.
7. Compares manual accuracy with sklearn accuracy.
8. Identifies misclassified students.
9. Compares testing accuracy using different random states.
10. Visualizes the Decision Tree.
11. Identifies the root node feature.
12. Creates a new `PerformanceIndex` feature.
13. Compares model performance after adding `PerformanceIndex`.
14. Calculates training and testing accuracy using `max_depth=None`.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Python Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
