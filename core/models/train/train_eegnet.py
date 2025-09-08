import os
import numpy as np
from keras.utils import to_categorical
from core.data.manager import DataManager
from core.models.eegnet_model import EEGNetModel

def train_eegnet(proc_id: str, model_name: str = "eegnet_eeg", num_classes: int = 2, epochs: int = 10):
    dm = DataManager()
    sample = dm.get_processed_sample(proc_id)

    X = np.expand_dims(sample.data, axis=-1)  # [n_channels, n_samples, 1]
    y = np.array(sample.labels)

    # Привести y к one-hot
    y_cat = to_categorical(y, num_classes=num_classes)

    # Разделим train/test (очень грубо для примера)
    split = int(0.8 * X.shape[0])
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_cat[:split], y_cat[split:]

    model = EEGNetModel(name=model_name, num_classes=num_classes)

    print("✅ Обучение EEGNet...")
    model.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

    # Куда сохранить
    out_dir = "storage/models/tensorflow"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{model_name}.h5")

    model.model.save(model_path)
    print(f"✅ EEGNet сохранена: {model_path}")


if __name__ == "__main__":
    # укажи ID обработанного датасета
    proc_id = "your_processed_id"
    train_eegnet(proc_id)
