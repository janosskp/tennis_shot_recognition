"""Train the tennis shot RNN on the local dataset, including myplayer.

This script mirrors the notebook training flow but runs in a standard Python
environment so it can use the working TensorFlow installation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.models import Sequential


DATASET_FOLDERS = [
    "nadal",
    "djoko_sock",
    "federer",
    "alcaraz",
    "dimitrov_alcaraz",
    "dimitrov_thiem",
    "roland",
    "myplayer",
]


def load_shots(dataset_root: Path):
    X = []
    y = []

    for folder in DATASET_FOLDERS:
        shots_dir = dataset_root / folder / "shots"
        if not shots_dir.exists():
            print(f"{shots_dir} does not exist, skipping")
            continue

        print(f"Loading shots from {shots_dir}")
        for shot_csv in sorted(shots_dir.iterdir()):
            if shot_csv.suffix.lower() != ".csv":
                continue

            data = pd.read_csv(shot_csv)

            # Normalize left-handed forehands from nadal to the same reference frame.
            if folder == "nadal":
                revert_data = data.copy()
                for feature in data.columns:
                    if feature.endswith("_x"):
                        revert_data[feature] = 1 - data[feature]
                data = revert_data

            features = data.loc[:, data.columns != "shot"]
            X.append(features.to_numpy())
            y.append(data["shot"].iloc[0])

    if not X:
        raise RuntimeError("No training samples were found.")

    X = np.stack(X, axis=0)
    y = np.array(y)
    return X, y


def encode_labels(labels):
    classes = np.unique(labels)
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = np.array([class_to_index[label] for label in labels], dtype=np.int32)
    return encoded, classes


def split_data(X, y, test_size=0.33, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    split_index = int(len(indices) * (1 - test_size))
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]


def main():
    dataset_root = Path("dataset")

    print("Loading shot CSVs...", flush=True)
    X, y = load_shots(dataset_root)
    print(f"Loaded {len(y)} shots for training")

    print("Encoding labels and splitting data...", flush=True)
    y_encoded, classes = encode_labels(y)
    X_train, X_val, y_train, y_val = split_data(X, y_encoded, test_size=0.33, random_state=42)

    print(f"Shape of train features: {X_train[0].shape}")
    print(f"Shape of val features: {X_val[0].shape}")

    nb_cat = len(classes)
    print("Categories:", list(classes))

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=nb_cat)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=nb_cat)

    print("Building model...", flush=True)
    model = Sequential(
        [
            GRU(units=24, dropout=0.1, input_shape=(30, 26)),
            Dropout(0.2),
            Dense(units=8, activation="relu"),
            Dense(units=nb_cat, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    checkpoint_path = "weights_myplayer.hdf5"
    callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, verbose=0, save_best_only=True),
        EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True),
    ]

    print("Starting training...", flush=True)
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        batch_size=32,
        epochs=60,
        verbose=1,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"Validation accuracy before checkpoint reload: {accuracy:.4f}")

    model.load_weights(checkpoint_path)
    loss, accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"Validation accuracy after checkpoint reload: {accuracy:.4f}")

    output_path = "tennis_rnn_myplayer.h5"
    model.save(output_path)
    print(f"Saved trained model to {output_path}")

    # Keep the history around for quick inspection if run interactively.
    return history


if __name__ == "__main__":
    main()