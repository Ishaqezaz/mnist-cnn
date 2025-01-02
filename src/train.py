from src.loader import Load
from src.augmenter import Augmenter
from src.model import MNISTModel
import yaml
import tensorflow as tf

def train():
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    learning_rate = config["training"]["learning_rate"]
    phase1_epochs = config["training"]["phase1_epochs"]
    phase2_epochs = config["training"]["phase2_epochs"]
    validation_steps = config["training"]["validation_steps"]
    steps_per_epoch = config["training"]["steps_per_epoch"]
    model_name = config["training"]["model_name"]

    # Data preparation
    loader = Load(config)
    (x_train, y_train), (x_test, y_test) = loader.load_data()

    augmenter = Augmenter(config)
    train_gen = augmenter.augment(x_train, y_train)
    val_gen = augmenter.augment(x_test, y_test)

    # Model
    model = MNISTModel(config)
    model.model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    # Phase 1: Normal training
    history_phase1 = model.model.fit(
        train_gen,
        validation_data = val_gen,
        epochs = phase1_epochs,
        validation_steps = len(x_test) // 32,
    )

    # Phase 2: Fine-tuning with reduced steps 
    history_phase2 = model.model.fit(
        train_gen,
        validation_data = val_gen,
        epochs = phase2_epochs,
        steps_per_epoch = steps_per_epoch,
        validation_steps = validation_steps,
    )
    
    model.save(model_name)

