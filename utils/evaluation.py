import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    root_mean_squared_error, 
    f1_score, 
    confusion_matrix
)

def evaluate_model(y_true, y_pred, metrics:dict[str,list], reports:list[str|dict]):
    acc = accuracy_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    metrics["acc"].append(acc)
    metrics["RMSE"].append(rmse)
    metrics["f1_macro"].append(f1_macro)
    metrics["f1_micro"].append(f1_micro)
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    reports.append(report)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    