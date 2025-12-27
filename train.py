import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd



# GPU SETTINGS
# =========================


from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")  

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set memory growth:", e)


# 1) CONFIG 
# =========================
IMG_SIZE = (224, 224)  
BATCH_SIZE = 64
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE



DATASET_ROOT = "/mnt/c/Users/hadee/Downloads/Brain Tumor Data Set/Brain_Tumor_Data_Set"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
TEST_DIR  = os.path.join(DATASET_ROOT, "test")

# Your counts (training set)
train_no_tumor = 1587
train_tumor    = 2013

total = train_no_tumor + train_tumor
class_weight = {
    0: total / (2.0 * train_no_tumor),  # no_tumor
    1: total / (2.0 * train_tumor)      # tumor
}
print("Class weights:", class_weight)

# =========================
# 2) LOAD DATA (GRAYSCALE)
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="training"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="validation"
)

# IMPORTANT: shuffle=False for correct evaluation
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Class order:", class_names)

# Speed pipeline (cache may use RAM; if RAM low remove cache())
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# =========================
# 3) HELPERS (NO LABEL MISMATCH)
# =========================
def get_true_labels(ds):
    return np.concatenate([y.numpy() for _, y in ds], axis=0).astype(int)

def evaluate_model(model, ds, title="Model", threshold=0.5):
    y_true = get_true_labels(ds)
    y_prob = model.predict(ds, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n===== {title} =====")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n",
          classification_report(y_true, y_pred, target_names=class_names, digits=4))
    return cm

def get_metrics(model, ds):
    vals = model.evaluate(ds, verbose=0)
    return dict(zip(model.metrics_names, vals))

# =========================
# 4) AUGMENTATION (LIGHT = FAST)
# =========================
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05),
], name="augmentation")


# 5) MODEL B — RESNET50 TRANSFER
# =========================
base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,)
)
base.trainable = False

inp = keras.Input(shape=IMG_SIZE + (1,))
x = data_aug(inp)
x = layers.Concatenate()([x, x, x])  # grayscale -> RGB
x = keras.applications.resnet.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

resnet_model = keras.Model(inp, out, name="ResNet50_Transfer")

resnet_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]
)

callbacks_resnet = [
    keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint("best_resnet50.keras", monitor="val_auc", mode="max", save_best_only=True),
]

print("\nTraining ResNet50 (frozen backbone)...")
resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks_resnet,
    class_weight=class_weight,
    verbose=1
)

print("\nFine-tuning ResNet50 top layers...")
base.trainable = True
fine_tune_at = 140
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

resnet_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]
)

resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=6,
    callbacks=callbacks_resnet,
    class_weight=class_weight,
    verbose=1
)

resnet_model = keras.models.load_model("best_resnet50.keras")
evaluate_model(resnet_model, test_ds, title="ResNet50 Transfer")

# =========================
# 7) COMPARISON TABLE 
# =========================
custom_metrics = get_metrics(custom_model, test_ds)
resnet_metrics = get_metrics(resnet_model, test_ds)

comparison = pd.DataFrame([
    {"Model": "Custom CNN",
     "Accuracy": custom_metrics["accuracy"],
     "Precision": custom_metrics["precision"],
     "Recall": custom_metrics["recall"],
     "AUC": custom_metrics["auc"]},
    {"Model": "ResNet50 Transfer",
     "Accuracy": resnet_metrics["accuracy"],
     "Precision": resnet_metrics["precision"],
     "Recall": resnet_metrics["recall"],
     "AUC": resnet_metrics["auc"]},
])

print("\n=== Comparison Table (TEST SET) ===")
print(comparison.to_string(index=False))

comparison.to_csv("comparison_table.csv", index=False)
print("\nSaved: comparison_table.csv")
