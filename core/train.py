import pandas as pd
import mlflow
import optuna

from .datasets import load_selected_dataset
from .utils import prepare_labeled_data, show_confusion_matrix
from .models_fabric import get_models

from datetime import datetime
from time import time
from os import path, makedirs

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

le = LabelEncoder()

def start_training(dataset_list, dataset_res, cv, optuna_settings):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_artifact_path = path.join("temp_plots", timestamp)
    makedirs(base_artifact_path, exist_ok=True)
    
    for dataset_id in dataset_list:
        with mlflow.start_run(run_name=f"Dataset_{dataset_id}", nested=True):
            print(f"\n--- Обработка Датасета {dataset_id} ---")
            raw = load_selected_dataset(dataset_id)
            sfreq = int(raw.info['sfreq'])
            models = get_models(sfreq)
            X, y = prepare_labeled_data(raw, dataset_id)
            
            y_encoded = le.fit_transform(y)
            unique_labels = le.classes_.astype(str)
            label_mapping = {'101': 'Audio', '102': 'Visual', '201': 'Easy', '202': 'Hard', '1': 'Relax', '2': 'Left', '3': 'Right'}
            display_labels = [label_mapping.get(l, l) for l in unique_labels]

            for name, proto in models.items():
                with mlflow.start_run(run_name=f"{name}_DS{dataset_id}", nested=True):
                    print(f"Оптимизация Optuna: {name}")

                    def objective(trial):
                        model = clone(proto)
                        p = {}
                        
                        current_model_settings = optuna_settings.get(name, {})
                        
                        for param_name, config in current_model_settings.items():
                            type_ = config[0]
                            
                            if type_ == "int":
                                p[param_name] = trial.suggest_int(param_name, config[1], config[2])
                            elif type_ == "float":
                                p[param_name] = trial.suggest_float(param_name, config[1], config[2])
                            elif type_ == "categorical":
                                p[param_name] = trial.suggest_categorical(param_name, config[1])
                                
                        model.set_params(**p)
                        return cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1).mean()

                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=15)
                    
                    best_model = clone(proto)
                    best_model.set_params(**study.best_params)
                    
                    start_time = time()
                    y_pred = cross_val_predict(best_model, X, y_encoded, cv=cv)
                    best_model.fit(X, y_encoded)
                    duration = time() - start_time
                    
                    acc = accuracy_score(y_encoded, y_pred)
                    f1 = f1_score(y_encoded, y_pred, average='macro', zero_division=0)

                    mlflow.log_param("model_name", name)
                    mlflow.log_param("dataset_id", dataset_id)
                    mlflow.log_param("sfreq", sfreq)
                    mlflow.log_params(study.best_params)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_macro", f1)
                    mlflow.log_metric("training_duration", duration)

                    show_confusion_matrix(y_encoded, y_pred, display_labels, name, dataset_id, base_artifact_path)

                    dataset_res.append({
                        'Dataset': dataset_id, 'Model': name, 'Accuracy': acc,
                        'F1_Macro': f1, 'Time': round(duration, 2), 'Params': str(study.best_params)
                    })
                    
    return pd.DataFrame(dataset_res)