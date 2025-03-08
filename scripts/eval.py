import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from absl import logging
import gin

from src.models import model_lib
from src.data import data_lib
from scripts import eval_lib
from src.utils import metrics_lib
from scripts import train_lib

def main():
    # Paths for the professor's system
    BASE_FOLDER = 'D:/azadegan/frame-interpolation/Checkpoint'
    LABEL = 'run0'
    
    # Use the local saved_model directory for portability
    LOCAL_MODEL_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_model')
    
    EVAL_FOLDER = os.path.join(BASE_FOLDER, LABEL, 'eval')
    # Use a fallback mechanism for SAVED_MODEL_FOLDER
    SAVED_MODEL_FOLDER = LOCAL_MODEL_FOLDER if os.path.exists(LOCAL_MODEL_FOLDER) else os.path.join(BASE_FOLDER, LABEL, 'saved_model')
    
    GIN_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'azadegan_net-Style.gin')
    if not os.path.exists(GIN_CONFIG):
        # Fallback to professor's expected path
        GIN_CONFIG = 'D:/azadegan/frame-interpolation/US/config/azadegan_net-Style.gin'
    
    # Create output folder if it doesn't exist
    os.makedirs(EVAL_FOLDER, exist_ok=True)
    
    # Log paths for debug purposes
    logging.info(f"BASE_FOLDER: {BASE_FOLDER}")
    logging.info(f"EVAL_FOLDER: {EVAL_FOLDER}")
    logging.info(f"SAVED_MODEL_FOLDER: {SAVED_MODEL_FOLDER}")
    logging.info(f"GIN_CONFIG: {GIN_CONFIG}")
    
    # Parse the gin configuration file
    logging.info(f"Parsing configuration from: {GIN_CONFIG}")
    gin.parse_config_files_and_bindings(
        config_files=[GIN_CONFIG],
        bindings=None,
        skip_unknown=True
    )
    
    # Setup distribution strategy (CPU or GPU)
    strategy_name = 'gpu' if tf.config.list_physical_devices('GPU') else 'cpu'
    strategy = train_lib.get_strategy(strategy_name)
    logging.info(f"Using distribution strategy: {strategy} ({strategy_name})")
    
    with strategy.scope():
        try:
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
        
        logging.info("Creating evaluation metrics")
        metrics = metrics_lib.create_metrics_fn()
    
    # Create evaluation datasets from the configuration
    logging.info("Creating evaluation datasets")
    try:
        eval_datasets = data_lib.create_eval_datasets() or None
        if eval_datasets:
            logging.info(f"Successfully created evaluation datasets")
        else:
            logging.warning("No evaluation datasets were created from gin config")
    except Exception as e:
        logging.error(f"Error creating evaluation datasets: {e}")
        eval_datasets = None
    
    # Create a summary writer for TensorBoard
    eval_summary_writer = tf.summary.create_file_writer(EVAL_FOLDER)
    CHECKPOINT_STEP = 0
    
    if eval_datasets:
        logging.info("Starting model evaluation...")
        
        try:
            eval_lib.eval_loop(
                strategy=strategy,
                eval_base_folder=EVAL_FOLDER,
                model=model,
                metrics=metrics,
                datasets=eval_datasets,
                summary_writer=eval_summary_writer,
                checkpoint_step=CHECKPOINT_STEP
            )
            
            logging.info("Model evaluation completed successfully.")
            logging.info(f"Results saved to: {EVAL_FOLDER}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
    else:
        logging.warning("Skipping evaluation as no datasets were created")

if __name__ == '__main__':
    main()
