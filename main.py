import os
import sys
import shutil
from pathlib import Path
from train_model import trainer
from inference import run_inference
from omegaconf import OmegaConf

num_args = len(sys.argv)
if num_args > 2:
    sys.exit(f"Too many arguments: Expected at most 1, got {num_args - 1}")
if num_args < 2:
    sys.exit(f"Expected at least 1 argument\nUsage:\npython main.py <mode: train | inference>")
if num_args == 2:
    if sys.argv[1] not in ['train', 'inference']:
        sys.exit(f"Mode '{sys.argv[1]}' is invalid!")

mode = sys.argv[1]

if mode == 'train':
    config = 'train_config.yaml'  # Name of the config file

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the config file
    fn = os.path.join(script_dir, config)

    # Load the config file using OmegaConf
    config = OmegaConf.load(fn)
    OmegaConf.set_readonly(config, True)  # Make the config immutable

    # Create the 'Outputs' directory in the same directory as the script
    outputs_path = os.path.join(script_dir, 'Outputs')
    os.makedirs(outputs_path, exist_ok=True)

    # Call the trainer function with the loaded config
    trainer(config=config)

if mode == 'inference':
    config = 'inference_config.yaml'

    parent = str(Path(__file__)).rsplit('\\', maxsplit=1)[0]
    fn = os.path.join(parent, config)
    config = OmegaConf.load(fn)  # Load the config directly with OmegaConf
    OmegaConf.set_read_only(config, True)  # Make the config immutable

    checkpoint_path = os.path.join(parent, 'Saved Checkpoints')

    # Create the 'Saved Checkpoints' directory if it doesn't exist
    os.makedirs(checkpoint_path, exist_ok=True)

    # List all .pth files in the checkpoint directory
    models = [i for i in os.listdir(checkpoint_path) if i.endswith('.pth')]

    # Check if there are any pre-trained models
    if len(models) < 1:
        sys.exit(f"No pre-trained models available. Train a model first.")


    
    # Check if the 'data/Train' directory exists, and create it if it doesn't
    data_train_path = os.path.join('data', 'Train')
    os.makedirs(data_train_path, exist_ok=True)

    # Check if the specified model exists
    model_name = f"{config.model_name}.pth"
    if model_name not in models:
        sys.exit(f"No model available with name: {model_name}")

    # Remove the file extension from the model name
    model_name = model_name.rsplit('.', maxsplit=1)[0]

    # Create the 'Outputs' directory if it doesn't exist
    outputs_path = os.path.join('Outputs')
    os.makedirs(outputs_path, exist_ok=True)

    # Create the inference log file
    logs = os.path.join(outputs_path, f'inference_logs_{model_name}.txt')
    output = open(logs, 'w', encoding='utf-8')

    # Run inference
    run_inference(model_name, output)

    # Close the log file
    output.close()