from pathlib import Path
import tensorflow as tf
from dvclive.keras import DVCLiveCallback
from dvc.api import params_show

# Set the paths to the train and validation directories
BASE_DIR = Path(__file__).parent.parent
data_dir = BASE_DIR / "data"

# Load the parameters from params.yaml
params = params_show()["train"]

##############################################################################
# Below, image width, height and batch_size parameters come from params.yaml #
##############################################################################

# Create an ImageDataGenerator object for the train set with augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    fill_mode="nearest",
)

train_generator = train_datagen.flow_from_directory(
    data_dir / "prepared" / "train",
    target_size=(params["image_width"], params["image_height"]),
    batch_size=params["batch_size"],
    class_mode="categorical",
)

# Do the same for test
test_dataget = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_dataget.flow_from_directory(
    data_dir / "prepared" / "test",
    target_size=(params["image_width"], params["image_height"]),
    batch_size=params["batch_size"],
    class_mode="categorical",
)


def get_model():
    """Define the model to be fit"""
    # Define a CNN model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                activation="relu",
                input_shape=(params["image_width"], params["image_height"], 3),
            ),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(43, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        # Learning rate is loaded from `params.yaml`
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    return model


def main():
    # Get the model
    model = get_model()
    # Create a path to save the model
    model_path = BASE_DIR / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path / "model.keras", monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5),
        DVCLiveCallback(dir="evaluation"),
    ]

    # Fit the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        # Number of epochs loaded from `params.yaml`
        epochs=params["n_epochs"],
        validation_data=test_generator,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
