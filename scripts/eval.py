import os
import tensorflow as tf
from absl import logging
import gin

from src.models import model_lib
from src.data import data_lib
from scripts import eval_lib
from src.utils import metrics_lib
from scripts import train_lib

def main():
    BASE_FOLDER = 'D:/azadegan/frame-interpolation/Checkpoint'
    LABEL = 'run0'
    
    EVAL_FOLDER = os.path.join(BASE_FOLDER, LABEL, 'eval')
    SAVED_MODEL_FOLDER = os.path.join(BASE_FOLDER, LABEL, 'saved_model')
    
    GIN_CONFIG = 'D:/azadegan/frame-interpolation/US/config/azadegan_net-Style.gin'
    gin.parse_config_files_and_bindings(
        config_files=[GIN_CONFIG],
        bindings=None,
        skip_unknown=True
    )
    
    strategy = train_lib.get_strategy('gpu')
    
    with strategy.scope():
        model = tf.keras.models.load_model(SAVED_MODEL_FOLDER, compile=False)
        metrics = metrics_lib.create_metrics_fn()
    
    eval_datasets = data_lib.create_eval_datasets() or None
    eval_summary_writer = tf.summary.create_file_writer(EVAL_FOLDER)
    CHECKPOINT_STEP = 0
    
    logging.info("Starting model evaluation...")
    
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

if __name__ == '__main__':
    main()
