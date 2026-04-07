import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost

def load_data():
    df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    
    # We drop 'Over18' because it's same for all rows, 'EmployeeCount', 'StandardHours', 'EmployeeNumber'
    df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'])
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
def run_experiment(name, model, X_train, X_test, y_train, y_test, is_xgb=False):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        f1 = f1_score(y_test, predictions)
        acc = accuracy_score(y_test, predictions)
        
        mlflow.log_params(model.get_params())
        mlflow.log_metric("validation_f1", f1)
        mlflow.log_metric("accuracy", acc)
        
        if is_xgb:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
            
        print(f"[{name}] F1 Score: {f1:.4f} | Accuracy: {acc:.4f}")
        return f1, mlflow.active_run().info.run_id

def main():
    if not os.path.exists("data/WA_Fn-UseC_-HR-Employee-Attrition.csv"):
        print("Data omitted. Please run `dvc pull`.")
        return

    # To use DagsHub, initialization automatically connects MLflow:
    import dagshub
    dagshub.init(repo_owner='JaxsonDyer', repo_name='AIDA2372A', mlflow=True)

    mlflow.set_experiment("Employee_Attrition_Prediction")
    
    X_train, X_test, y_train, y_test = load_data()
    
    runs = {}
    
    # 1. Logistic Regression Baseline
    print("Running Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    f1, run_id = run_experiment("Logistic_Regression_Baseline", lr, X_train, X_test, y_train, y_test)
    runs[run_id] = f1
    
    # 2. Random Forest (Default)
    print("Running Random Forest Default...")
    rf_default = RandomForestClassifier(random_state=42, class_weight='balanced')
    f1, run_id = run_experiment("Random_Forest_Default", rf_default, X_train, X_test, y_train, y_test)
    runs[run_id] = f1
    
    # 3. Random Forest (Tuned)
    print("Running Random Forest Tuned...")
    rf_tuned = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    f1, run_id = run_experiment("Random_Forest_Tuned", rf_tuned, X_train, X_test, y_train, y_test)
    runs[run_id] = f1
    
    # 4. XGBoost Classifier
    print("Running XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train))
    f1, run_id = run_experiment("XGBoost", xgb, X_train, X_test, y_train, y_test, is_xgb=True)
    runs[run_id] = f1
    
    # Find Best Model
    best_run_id = max(runs, key=runs.get)
    best_f1 = runs[best_run_id]
    print(f"\nBest Model Run ID: {best_run_id} with F1: {best_f1:.4f}")
    
    client = mlflow.tracking.MlflowClient()
    model_name = "EmployeeAttritionModel"
    
    # Register the best model
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, model_name)
    print(f"Registered model {model_name} from run {best_run_id}")

if __name__ == "__main__":
    main()
