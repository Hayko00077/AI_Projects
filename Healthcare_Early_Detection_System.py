# ---------------------------
# Diabetes Prediction Project with Multiple Models + Visualizations
# ---------------------------

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("diabetes.csv")
print("Dataset loaded successfully")
print(df.head())

# ---------------------------
# Store in SQLite
# ---------------------------
conn = sqlite3.connect("diabetes.db")
df.to_sql("diabetes", conn, if_exists="replace", index=False)
print("Dataset stored in SQLite")

# ---------------------------
# Data cleaning
# ---------------------------
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.nan)
df.fillna(df.median(), inplace=True)
print("Data cleaning completed")
print(df.isnull().sum())

# ---------------------------
# Visualizations: Feature distributions
# ---------------------------
plt.figure(figsize=(15,10))
df.hist(bins=20, edgecolor='black', linewidth=1.2, figsize=(15,10))
plt.suptitle("Feature Distributions", fontsize=20)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.show()

# ---------------------------
# Features and target
# ---------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------
# Scale features for some models
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Define models
# ---------------------------
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
et = ExtraTreesClassifier(n_estimators=200, random_state=42)
svm = SVC(probability=True, random_state=42)
cat = CatBoostClassifier(verbose=0, random_state=42, iterations=200)

xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_params = {'n_estimators':[100,200],'max_depth':[3,5],'learning_rate':[0.01,0.1]}
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='roc_auc', n_jobs=-1)

mlp_model = MLPClassifier(max_iter=1000, random_state=42)
mlp_params = {'hidden_layer_sizes':[(64,32),(128,64,32)],'activation':['relu','tanh'],'alpha':[0.0001,0.001]}
mlp_grid = GridSearchCV(mlp_model, mlp_params, cv=3, scoring='roc_auc', n_jobs=-1)

# ---------------------------
# Models dictionary
# ---------------------------
models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "ExtraTrees": et,
    "SVM": svm,
    "CatBoost": cat,
    "XGBoost": xgb_grid,
    "Neural Network": mlp_grid
}

results = {}

# ---------------------------
# Train and evaluate models
# ---------------------------
for name, model in models.items():
    print(f"\n=== {name} ===")
    if name in ["Logistic Regression","SVM","Neural Network"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    print(f"Accuracy: {acc:.3f} | ROC-AUC: {roc:.3f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    results[name] = {"model":model,"accuracy":acc,"roc_auc":roc}

# ---------------------------
# Voting ensemble (exclude CatBoost)
# ---------------------------
top_models = sorted([(name,res) for name,res in results.items() if name!="CatBoost"], key=lambda x: x[1]["roc_auc"], reverse=True)[:5]
voting_estimators = [(name,res["model"]) for name,res in top_models]

voting_clf = VotingClassifier(estimators=voting_estimators, voting="soft")
voting_clf.fit(X_train_scaled, y_train)

y_pred_vote = voting_clf.predict(X_test_scaled)
y_prob_vote = voting_clf.predict_proba(X_test_scaled)[:,1]

acc_vote = accuracy_score(y_test, y_pred_vote)
roc_vote = roc_auc_score(y_test, y_prob_vote)

print("\n=== Voting Ensemble ===")
print(f"Accuracy: {acc_vote:.3f} | ROC-AUC: {roc_vote:.3f}")
print(classification_report(y_test, y_pred_vote))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_vote))
results["Voting Ensemble"] = {"model":voting_clf,"accuracy":acc_vote,"roc_auc":roc_vote}

# ---------------------------
# Save best model
# ---------------------------
best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
best_model = results[best_model_name]["model"]
joblib.dump(best_model, "best_model.pkl")
print(f"\nBest model ({best_model_name}) saved based on ROC-AUC")

# ---------------------------
# Example single prediction
# ---------------------------
sample_row = X_test.iloc[[0]]
if best_model_name in ["Logistic Regression","SVM","Neural Network"]:
    prob = best_model.predict_proba(scaler.transform(sample_row))[:,1][0]
else:
    prob = best_model.predict_proba(sample_row)[:,1][0]

risk = "High Risk" if prob>0.5 else "Low Risk"
print({"Row":sample_row.index[0],"Risk":risk,"Probability":prob})

# ---------------------------
# Visualization: Model performance
# ---------------------------
model_names = list(results.keys())
accuracy = [results[m]["accuracy"] for m in model_names]
roc_auc = [results[m]["roc_auc"] for m in model_names]

plt.figure(figsize=(10,6))
sns.barplot(x=model_names, y=accuracy)
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=model_names, y=roc_auc)
plt.xticks(rotation=45)
plt.title("Model ROC-AUC Comparison")
plt.show()

# Confusion matrix heatmap for voting ensemble
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_vote), annot=True, fmt="d", cmap="Blues")
plt.title("Voting Ensemble Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
