import os
import argparse
from params import DEFAULT_TRAIN_DATA_PATH, DEFAULT_TEST_DATA_PATH, DEFAULT_MODEL_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Predict using the model with given data")
    parser.add_argument('--train-data', type=str, default=DEFAULT_TRAIN_DATA_PATH, help='Path to the training data csv file')
    parser.add_argument('--test-data', type=str, default=DEFAULT_TEST_DATA_PATH, help='Path to the test data csv file')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help='Path to the model h5 file')

    args = parser.parse_args()

    if args.train_data and not os.path.isfile(args.train_data):
        raise FileNotFoundError(f"The training file {args.train_data} does not exist.")

    if args.test_data and not os.path.isfile(args.test_data):
        raise FileNotFoundError(f"The test file {args.test_data} does not exist.")

    if args.model_path and not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"The model file {args.model_path} does not exist.")

    return args
