import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from absl import logging

# Import gin configuration system
import gin

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
    
    This is an alternative evaluation script that allows specifying custom
    evaluation datasets directly, rather than using the datasets defined in
    the gin configuration file.
    """

    # Paths for the professor's system
    BASE_FOLDER = 'D:/azadegan/frame-interpolation/Checkpoint'
    LABEL = 'run0'

    # Use the local saved_model directory for portability
    LOCAL_MODEL_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_model')
    
    # Define the evaluation and saved_model folders
    EVAL_FOLDER = os.path.join(BASE_FOLDER, LABEL, 'eval')
    # Use a fallback mechanism for SAVED_MODEL_FOLDER
    SAVED_MODEL_FOLDER = LOCAL_MODEL_FOLDER if os.path.exists(LOCAL_MODEL_FOLDER) else os.path.join(BASE_FOLDER, LABEL, 'saved_model')
    
    # Create output folder if it doesn't exist
    os.makedirs(EVAL_FOLDER, exist_ok=True)
    
    # Log paths for debug purposes
    logging.info(f"BASE_FOLDER: {BASE_FOLDER}")
    logging.info(f"EVAL_FOLDER: {EVAL_FOLDER}")
    logging.info(f"SAVED_MODEL_FOLDER: {SAVED_MODEL_FOLDER}")
    
    # Optionally parse a gin configuration
    # GIN_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'azadegan_net-Style.gin')
    # if not os.path.exists(GIN_CONFIG):
    #     # Fallback to professor's expected path
    #     GIN_CONFIG = 'D:/azadegan/frame-interpolation/US/config/azadegan_net-Style.gin'
    # 
    # logging.info(f"Parsing configuration from: {GIN_CONFIG}")
    # gin.parse_config_files_and_bindings(
    #     config_files=[GIN_CONFIG],
    #     bindings=None,
    #     skip_unknown=True
    # )

    # Choose a distribution strategy: 'gpu' for GPUs, 'cpu' for CPU, etc.
    strategy_name = 'gpu' if tf.config.list_physical_devices('GPU') else 'cpu'
    strategy = train_lib.get_strategy(strategy_name)
    logging.info(f"Using distribution strategy: {strategy} ({strategy_name})")

    with strategy.scope():
        try:
            # Load the saved model
            logging.info(f"Loading model from: {SAVED_MODEL_FOLDER}")
            model = tf.keras.models.load_model(SAVED_MODEL_FOLDER, compile=False)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            logging.info("Checking if model exists...")
            if os.path.exists(SAVED_MODEL_FOLDER):
                files = os.listdir(SAVED_MODEL_FOLDER)
                logging.info(f"Found files in model directory: {files}")
            else:
                logging.error(f"Model directory does not exist: {SAVED_MODEL_FOLDER}")
                return

        # Create evaluation metrics
        logging.info("Creating evaluation metrics")
        metrics = metrics_lib.create_metrics_fn()

    # Build the evaluation dataset using data_lib
    # These settings are for the professor's specific test video/dataset
    BATCH_SIZE = 4
    EVAL_FILES = ['D:/azadegan/frame-interpolation/datasets/eval_dataset@10']
    TRI_VIDEO_PATH = 'D:/azadegan/SF_Net-v1-main2/SF_Net-v1-main/Tri-00001-of-002'
    if os.path.exists(TRI_VIDEO_PATH):
        EVAL_FILES = [TRI_VIDEO_PATH]
        
    EVAL_NAMES = ['eval_dataset']
    CROP_SIZE = 256

    # Log dataset paths
    logging.info(f"Evaluation file paths: {EVAL_FILES}")
    
    # Try to find alternative dataset paths if the specified ones don't exist
    if not all(os.path.exists(f) for f in EVAL_FILES):
        logging.warning(f"Some dataset paths don't exist. Trying to find alternatives...")
        # Check if we can find the TFRecord in the current directory structure
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for root, dirs, files in os.walk(project_root):
            for file in files:
                if file.endswith('.tfrecord') or 'Tri-' in file:
                    alt_path = os.path.join(root, file)
                    logging.info(f"Found potential dataset: {alt_path}")
                    EVAL_FILES = [alt_path]
                    break
            if EVAL_FILES != ['D:/azadegan/frame-interpolation/datasets/eval_dataset@10']:
                break

    try:
        logging.info(f"Creating evaluation datasets from: {EVAL_FILES}")
        eval_datasets = data_lib.create_eval_datasets(
            batch_size=BATCH_SIZE,
            files=EVAL_FILES,
            names=EVAL_NAMES,
            crop_size=CROP_SIZE
        )
        
        if eval_datasets:
            logging.info(f"Successfully created {len(eval_datasets)} evaluation datasets")
        else:
            logging.warning("No evaluation datasets were created, check file paths")
    except Exception as e:
        logging.error(f"Error creating evaluation datasets: {e}")
        eval_datasets = None

    # Create a summary writer for logging evaluation results
    eval_summary_writer = tf.summary.create_file_writer(EVAL_FOLDER)

    # Define a checkpoint step for logging
    CHECKPOINT_STEP = 0

    if eval_datasets:
        logging.info("Starting model evaluation...")
        
        try:
            # Run the evaluation loop
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
            logging.info(f"Results saved to: {EVAL_FOLDER}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
    else:
        logging.warning("Skipping evaluation as no datasets were created")

if __name__ == '__main__':
    main()
