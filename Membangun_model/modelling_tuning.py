import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Load dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('Sales_Class', axis=1)
y_train = train_df['Sales_Class']
X_test = test_df.drop('Sales_Class', axis=1)
y_test = test_df['Sales_Class']

mlflow.set_experiment("ps4-sales-classification-tuning")

# Hyperparameter tuning loop (example: C = [0.01, 0.1, 1.0, 10.0])
for C_val in [0.01, 0.1, 1.0, 10.0]:
    with mlflow.start_run():
        model = LogisticRegression(C=C_val, max_iter=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Manual logging
        mlflow.log_param("C", C_val)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Simpan model
        mlflow.sklearn.log_model(model, "model")

        print(f"C={C_val} | Accuracy={acc:.4f} | F1={f1:.4f}")
