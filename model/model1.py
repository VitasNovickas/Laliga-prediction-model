import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.model_selection import GridSearchCV

df = pd.read_json(
    r'The file path of training data',
    orient='index'
)
current_season = pd.read_json(r'The file path of LaLiga 25/26 season data', orient='index')

FEATURES = [
        "Games",
        "SoT%",
        "Gsh",
        "GSoT",
        "Win_rate10",
        "Wstrk_10",
        "proxy_xG",
        "proxy_xGA",
        "G90",
        "S90",
        "SoT90",
        "C90",
        "AG90",
        "AS90",
        "ASoT90",
        "F90",
        "Y90",
        "R90",
        "season"
]

X_current = current_season[FEATURES]
X = df[FEATURES]
y= df["top1"]

X_tr = X[df['season'] < 20]
y_tr = y[df['season'] < 20]
X_te = X[df['season'] >= 20]
y_te = y[df['season'] >= 20]

neg = (y_tr == 0).sum()
pos = (y_tr == 1).sum()

model = XGBClassifier(
    scale_pos_weight=neg/pos,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
    'subsample': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5] 
}

grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid.fit(X_tr, y_tr)

print(f"Best params: {grid.best_params_}")
print(f"Test AUC: {roc_auc_score(y_te, grid.predict_proba(X_te)[:,1]):.3f}")


final_model = XGBClassifier(**grid.best_params_,
    scale_pos_weight=neg/pos,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
final_model.fit(X, y) 


current_season['top1_Prob'] = final_model.predict_proba(X_current)[:, 1]

final_model.save_model('model/laliga_top1_w10_model.json')
