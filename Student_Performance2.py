import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# ------------------------------------------------------------
# Display Border
# ------------------------------------------------------------

def Border():
    print("-" * 70)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

def LoadData(filename):
    Data = pd.read_csv(filename)
    return Data

# ------------------------------------------------------------
# Prepare Data
# ------------------------------------------------------------

def PrepareData(Data):
    Features = [
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsCompleted",
        "SleepHours"
    ]

    X = Data[Features]
    Y = Data["FinalResult"]

    return X, Y

# ------------------------------------------------------------
# Split Dataset
# ------------------------------------------------------------

def SplitData(X, Y, random_state=42):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=random_state
    )

    return X_train, X_test, Y_train, Y_test

# ------------------------------------------------------------
# Train Decision Tree
# ------------------------------------------------------------

def TrainModel(X_train, Y_train, max_depth=None, random_state=42):
    Model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )

    Model.fit(X_train, Y_train)

    return Model

# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------

def DisplayFeatureImportance(Model, X):
    Border()
    print("Feature Importance")
    Border()

    Importance = Model.feature_importances_

    Result = pd.DataFrame({
        "Feature": X.columns,
        "Importance": Importance
    })

    print(Result.to_string(index=False))

    MostImportant = X.columns[Importance.argmax()]
    LeastImportant = X.columns[Importance.argmin()]

    print("Most Important Feature :", MostImportant)
    print("Least Important Feature :", LeastImportant)

# ------------------------------------------------------------
# Compare Model Without SleepHours
# ------------------------------------------------------------

def CompareWithoutSleep(Data, X_train, X_test, Y_train, Y_test, OriginalAccuracy):
    Border()
    print("Model Without SleepHours")
    Border()

    Features = [
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsCompleted"
    ]

    X = Data[Features]

    X_train_new = X.loc[X_train.index]
    X_test_new = X.loc[X_test.index]

    Model = DecisionTreeClassifier(random_state=42)
    Model.fit(X_train_new, Y_train)

    Y_pred = Model.predict(X_test_new)

    NewAccuracy = accuracy_score(Y_test, Y_pred)

    print("Original Accuracy :", OriginalAccuracy)
    print("Accuracy Without SleepHours :", NewAccuracy)
    print("Accuracy Difference :", NewAccuracy - OriginalAccuracy)

# ------------------------------------------------------------
# Compare Selected Features
# ------------------------------------------------------------

def CompareSelectedFeatures(Data, X_train, X_test, Y_train, Y_test, OriginalAccuracy):
    Border()
    print("Model Using StudyHours and Attendance")
    Border()

    Features = [
        "StudyHours",
        "Attendance"
    ]

    X = Data[Features]

    X_train_new = X.loc[X_train.index]
    X_test_new = X.loc[X_test.index]

    Model = DecisionTreeClassifier(random_state=42)
    Model.fit(X_train_new, Y_train)

    Y_pred = Model.predict(X_test_new)

    NewAccuracy = accuracy_score(Y_test, Y_pred)

    print("Full Feature Accuracy :", OriginalAccuracy)
    print("StudyHours and Attendance Accuracy :", NewAccuracy)

# ------------------------------------------------------------
# Predict New Students
# ------------------------------------------------------------

def PredictNewStudents(Model):
    Border()
    print("Prediction for New Students")
    Border()

    NewStudents = pd.DataFrame({
        "StudyHours": [4, 6, 8, 3, 7],
        "Attendance": [75, 85, 92, 65, 88],
        "PreviousScore": [55, 66, 78, 45, 72],
        "AssignmentsCompleted": [6, 7, 9, 4, 8],
        "SleepHours": [7, 7, 8, 6, 7]
    })

    Predictions = Model.predict(NewStudents)

    NewStudents["Prediction"] = Predictions

    NewStudents["Result"] = NewStudents["Prediction"].map({
        1: "Pass",
        0: "Fail"
    })

    print(NewStudents.to_string(index=False))

# ------------------------------------------------------------
# Manual Accuracy Calculation
# ------------------------------------------------------------

def CalculateManualAccuracy(Y_test, Y_pred, SklearnAccuracy):
    Border()
    print("Manual Accuracy Calculation")
    Border()

    Correct = 0
    Total = len(Y_test)

    for Actual, Predicted in zip(Y_test, Y_pred):
        if Actual == Predicted:
            Correct = Correct + 1

    ManualAccuracy = Correct / Total

    print("Correct Predictions :", Correct)
    print("Total Predictions :", Total)
    print("Manual Accuracy :", ManualAccuracy)
    print("Sklearn Accuracy :", SklearnAccuracy)
    print("Difference :", ManualAccuracy - SklearnAccuracy)

# ------------------------------------------------------------
# Display Misclassified Students
# ------------------------------------------------------------

def DisplayMisclassifiedStudents(X_test, Y_test, Y_pred):
    Border()
    print("Misclassified Students")
    Border()

    Result = X_test.copy()
    Result["ActualResult"] = Y_test
    Result["PredictedResult"] = Y_pred

    Misclassified = Result[Result["ActualResult"] != Result["PredictedResult"]]

    if len(Misclassified) == 0:
        print("No students were misclassified.")
        return

    print(Misclassified.to_string())

    print("Number of Misclassified Students :", len(Misclassified))
    print("Average StudyHours :", Misclassified["StudyHours"].mean())
    print("Average Attendance :", Misclassified["Attendance"].mean())
    print("Average PreviousScore :", Misclassified["PreviousScore"].mean())
    print("Average AssignmentsCompleted :", Misclassified["AssignmentsCompleted"].mean())
    print("Average SleepHours :", Misclassified["SleepHours"].mean())
    print("The averages above can be used to observe common patterns among misclassified students.")

# ------------------------------------------------------------
# Compare Random States
# ------------------------------------------------------------

def CompareRandomStates(X, Y):
    Border()
    print("Comparison of Random States")
    Border()

    RandomStates = [0, 10, 42]

    for State in RandomStates:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=State
        )

        Model = DecisionTreeClassifier(random_state=42)
        Model.fit(X_train, Y_train)

        Y_pred = Model.predict(X_test)

        Accuracy = accuracy_score(Y_test, Y_pred)

        print("Random State :", State, "Testing Accuracy :", Accuracy)

# ------------------------------------------------------------
# Visualize Decision Tree
# ------------------------------------------------------------

def DisplayDecisionTree(Model, X):
    Border()
    print("Decision Tree Visualization")
    Border()

    plt.figure(figsize=(18, 10))

    plot_tree(
        Model,
        feature_names=X.columns,
        class_names=["Fail", "Pass"],
        filled=True
    )

    plt.title("Student Performance Decision Tree")
    plt.show()

    RootFeatureIndex = Model.tree_.feature[0]

    if RootFeatureIndex != -2:
        RootFeature = X.columns[RootFeatureIndex]
        print("Root Node Feature :", RootFeature)
        print("The root feature is selected because it provides the best split according to the Decision Tree algorithm.")

# ------------------------------------------------------------
# Performance Index
# ------------------------------------------------------------

def ComparePerformanceIndex(Data, Y):
    Border()
    print("Model With PerformanceIndex")
    Border()

    NewData = Data.copy()

    NewData["PerformanceIndex"] = (NewData["StudyHours"] * 2) + NewData["Attendance"]

    Features = [
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsCompleted",
        "SleepHours",
        "PerformanceIndex"
    ]

    X = NewData[Features]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    Model = DecisionTreeClassifier(random_state=42)
    Model.fit(X_train, Y_train)

    Y_pred = Model.predict(X_test)

    Accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy With PerformanceIndex :", Accuracy)

# ------------------------------------------------------------
# Training and Testing Accuracy
# ------------------------------------------------------------

def DisplayTrainingTestingAccuracy(X_train, X_test, Y_train, Y_test):
    Border()
    print("Training and Testing Accuracy")
    Border()

    Model = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    Model.fit(X_train, Y_train)

    TrainingPrediction = Model.predict(X_train)
    TestingPrediction = Model.predict(X_test)

    TrainingAccuracy = accuracy_score(Y_train, TrainingPrediction)
    TestingAccuracy = accuracy_score(Y_test, TestingPrediction)

    print("Training Accuracy :", TrainingAccuracy)
    print("Testing Accuracy :", TestingAccuracy)

    if TrainingAccuracy == 1.0 and TestingAccuracy < TrainingAccuracy:
        print("The model may be overfitting because it has learned the training data very closely and performs lower on unseen testing data.")
    else:
        print("Training and testing accuracy should be compared to understand how the model performs on seen and unseen data.")

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------

def main():
    Border()
    print("Student Performance Machine Learning")
    Border()

    Data = LoadData("student_performance_ml.csv")

    print("Dataset Loaded Successfully")
    print("Dataset Shape :", Data.shape)

    X, Y = PrepareData(Data)

    X_train, X_test, Y_train, Y_test = SplitData(X, Y, random_state=42)

    Model = TrainModel(X_train, Y_train, max_depth=None, random_state=42)

    Y_pred = Model.predict(X_test)

    Accuracy = accuracy_score(Y_test, Y_pred)

    Border()
    print("Original Decision Tree Model")
    Border()

    print("Testing Accuracy :", Accuracy)

    DisplayFeatureImportance(Model, X)
    CompareWithoutSleep(Data, X_train, X_test, Y_train, Y_test, Accuracy)
    CompareSelectedFeatures(Data, X_train, X_test, Y_train, Y_test, Accuracy)
    PredictNewStudents(Model)
    CalculateManualAccuracy(Y_test, Y_pred, Accuracy)
    DisplayMisclassifiedStudents(X_test, Y_test, Y_pred)
    CompareRandomStates(X, Y)
    DisplayDecisionTree(Model, X)
    ComparePerformanceIndex(Data, Y)
    DisplayTrainingTestingAccuracy(X_train, X_test, Y_train, Y_test)

    Border()
    print("Assignment 40 Completed Successfully")
    Border()

if __name__ == "__main__":
    main()