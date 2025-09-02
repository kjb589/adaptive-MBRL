import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from attention_mechanism import AttentionMechanism
from data_loader import DataLoader
import os


# Is it ready
# Yes


# Parameters
MODE = 1    # 0 for train, 1 for retrain
BATCH_SIZE = 8
EPOCHS = 10
SEQ_LEN = 8
IMAGE_SIZE = (480, 270, 3)
LEARNING_RATE = 1e-4
DATA_ROOT = 'C:/Users/kjbut/MAVS-Examples/SPAV/nuScenes/'
MODEL_PATH = 'C:/Users/kjbut/MAVS-Examples/models/attention_mechanism.keras'

# Initialize loader and get datasets
loader = DataLoader(DATA_ROOT, sequence_length=SEQ_LEN, image_size=(128, 128))
train_videos, train_labels = loader.get_sequences_and_labels()

# Instantiate model
if MODE == 0:
    model = AttentionMechanism(in_channels=3, embed_dim=128, num_heads=4)
elif MODE == 1:
    model = load_model(filepath=MODEL_PATH, custom_objects={"AttentionMechanism": AttentionMechanism})
else:
    print(ValueError, "Training mode not set or invalid value")
    exit(1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss={"weight": "binary_crossentropy"},
    metrics={"weight": "accuracy"}
)

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor=['loss','val_loss'], verbose=0),
    ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True)
]

# Train model
model.fit(x=train_videos, y={"weight": train_labels}, epochs=EPOCHS, callbacks=callbacks, verbose=2)
