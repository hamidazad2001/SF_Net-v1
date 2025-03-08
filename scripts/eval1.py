import os
import tensorflow as tf
from absl import logging

# If you need to parse a gin config file for dataset or model parameters,
# uncomment and configure the following line:
# import gin

from src.models import model_lib
from src.data import data_lib
from scripts import eval_lib
from src.utils import metrics_lib
from scripts import train_lib  # Used for get_strategy (cpu/gpu)

def main():
    """
    This script evaluates a saved TensorFlow model using the existing framework.
    It loads the model from the 'saved_model' directory, creates an evaluation
    dataset, and runs the eval_loop from eval_lib to measure performance.

    Make sure to set the correct paths for:
    - BASE_FOLDER, LABEL
    - The dataset TFRecord paths
    - The distribution strategy (cpu/gpu)
    """

    
    BASE_FOLDER = 'D:/azadegan/frame-interpolation/Checkpoint'
    LABEL = 'run0'

    # 2. Define the evaluation and saved_model folders
    EVAL_FOLDER = os.path.join(BASE_FOLDER, LABEL, 'eval')
    SAVED_MODEL_FOLDER = os.path.join(BASE_FOLDER, LABEL, 'saved_model')

    
    # 4. Choose a distribution strategy: 'gpu' for GPUs, 'cpu' for CPU, etc.
    strategy = train_lib.get_strategy('gpu')

    with strategy.scope():
        # 5. Load the saved model. If your model has custom layers, 
        # you might need custom_objects in load_model(...)
        model = tf.keras.models.load_model(SAVED_MODEL_FOLDER, compile=False)

        # 6. Create evaluation metrics from metrics_lib
        metrics = metrics_lib.create_metrics_fn()

    # 7. Build the evaluation dataset using data_lib
    # Update the following to match your actual dataset paths, batch size, and crop size
    BATCH_SIZE = 4
    EVAL_FILES = ['D:/azadegan/frame-interpolation/datasets/eval_dataset@10']
    EVAL_NAMES = ['eval_dataset']
    CROP_SIZE = 256

    eval_datasets = data_lib.create_eval_datasets(
        batch_size=BATCH_SIZE,
        files=EVAL_FILES,
        names=EVAL_NAMES,
        crop_size=CROP_SIZE
    )

    # 8. Create a summary writer for logging evaluation results
    eval_summary_writer = tf.summary.create_file_writer(EVAL_FOLDER)

    # 9. Optionally, define a checkpoint step for logging
    CHECKPOINT_STEP = 0

    logging.info("Starting model evaluation...")

    # 10. Run the evaluation loop
    eval_lib.eval_loop(
        strategy=strategy,
        eval_base_folder=EVAL_FOLDER,
        model=model,
        metrics=metrics,
        datasets=eval_datasets,
        summary_writer=eval_summary_writer,
        checkpoint_step=CHECKPOINT_STEP
    )

    logging.info("Model evaluation has completed successfully.")

if __name__ == '__main__':
    main()
