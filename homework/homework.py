import os
import pandas as pd
import gzip
import pickle
import json

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Rutas 

dir_input_test = "files/input/test_data.csv.zip"
dir_input_train = "files/input/train_data.csv.zip"
dir_models = "files/models/model.pkl.gz"
dir_metrics = "files/output/metrics.json"

# Paso 1: Cargar y limpiar datos

def load_data():
    df_train = pd.read_csv(dir_input_train, compression='zip')
    df_test = pd.read_csv(dir_input_test, compression='zip')
    return df_train, df_test

def clean_data(df):
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])

    df = df[df['EDUCATION'] != 0]
    df = df[df['MARRIAGE'] != 0]
    
    df = df.dropna()

    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    
    return df

df_train, df_test = load_data()
df_train = clean_data(df_train)
df_test = clean_data(df_test)

print("Paso 1: Datos cargados y limpiados")
print(f"{df_train.head()}\n\n{df_test.head()}")


# Paso 2: Separar en x_train, y_train, x_test, y_test

def split_data(df_train, df_test):
    x_train = df_train.drop(columns=["default"])
    y_train = df_train["default"]
    x_test = df_test.drop(columns=["default"])
    y_test = df_test["default"]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = split_data(df_train, df_test)
print("Paso 2: Datos separados")


# Paso 3: Crear pipeline de modelo

categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE',
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
numeric_cols = [c for c in x_train.columns if c not in categorical_cols]
preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols),],
    remainder='drop')

def create_pipeline():
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    return pipeline

pipeline = create_pipeline()

print("Paso 3: Pipeline creado")


# Paso 4: Optimizar hiperparámetros

def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {
        'rf__n_estimators': [400, 600], 
        'rf__max_depth': [20, 30],      
        'rf__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10, 
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(x_train, y_train)
    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor balanced_accuracy:", grid_search.best_score_)
    return grid_search

best_pipeline = optimize_hyperparameters(pipeline, x_train, y_train)
print("Paso 4: Hiperparámetros optimizados")


# Paso 5: Guardar modelo

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(model, f)

save_model(best_pipeline,  dir_models)
print(f"Paso 5: Modelo guardado en {dir_models}")


# Paso 6: Calcular métricas y guardar en metrics.json

def calculate_metrics(model, x, y, dataset_type):
    y_pred = model.predict(x)
    precision = precision_score(y, y_pred, zero_division=0) 
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y, y_pred)

    return {
        'type': 'metrics', 
        'dataset': dataset_type,
        'precision': round(precision, 4),
        'balanced_accuracy': round(balanced_acc, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }

metrics = []
metrics.append(calculate_metrics(best_pipeline, x_train, y_train, 'train'))
metrics.append(calculate_metrics(best_pipeline, x_test, y_test, 'test'))

os.makedirs(os.path.dirname(dir_metrics), exist_ok=True)
with open(dir_metrics, 'w') as f:
    for metric in metrics:
        f.write(json.dumps(metric) + '\n')

print(f"Paso 6: Métricas guardadas en {dir_metrics}")


# Paso 7: Matrices de confusión

def calculate_confusion_matrix(model, x, y, dataset_type):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)

    return {
        'type': 'cm_matrix',
        'dataset': dataset_type,
        'true_0': {
            'predicted_0': int(cm[0, 0]),
            'predicted_1': int(cm[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm[1, 0]),
            'predicted_1': int(cm[1, 1])
        }
    }

cm_train = calculate_confusion_matrix(best_pipeline, x_train, y_train, 'train')
cm_test = calculate_confusion_matrix(best_pipeline, x_test, y_test, 'test')

with open(dir_metrics, 'a') as f:
    f.write(json.dumps(cm_train) + '\n')
    f.write(json.dumps(cm_test) + '\n')

print(f"Paso 7: Matrices de confusión guardadas en {dir_metrics}")
print("Proceso completado.")