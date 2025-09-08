import lightgbm as lgb
import joblib
import os
import numpy as np
from core.data.manager import DataManager

def train_lightgbm(feat_id: str, model_name: str = "lightgbm_eeg"):
    dm = DataManager()
    X, y = dm.get_features_data(feat_id)

    # Датасет для LightGBM
    train_data = lgb.Dataset(X, label=y)

    params = {
        "objective": "binary",   # если у тебя классификация 2 классов
        "metric": "accuracy",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
    }

    print("✅ Обучение LightGBM...")
    gbm = lgb.train(params, train_data, num_boost_round=100)

    # Куда сохранить
    out_dir = "storage/models/lightgbm"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{model_name}.pkl")

    joblib.dump(gbm, model_path)
    print(f"✅ Модель сохранена: {model_path}")


if __name__ == "__main__":
    # укажи ID фичей из базы
    feat_id = "your_features_id"
    train_lightgbm(feat_id)
