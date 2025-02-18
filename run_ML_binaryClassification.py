import logging
import sys
import argparse
import json
from dataHandler import Config, OligosHandler, MetadataHandler, FeatureManager
from modelEvaluator import PredictionManager

def run_binaryClassification(config_file, filter_metadata_internal=None):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        stream=sys.stdout
    )

    # Initialize the configuration.
    config = Config(config_file=config_file)

    # If an external set is provided
    if filter_metadata_internal:
        # Process external set
        metadata_handler = MetadataHandler(config)
        oligos_handler = OligosHandler(config)
        feature_manager = FeatureManager(config, metadata_handler, oligos_handler, with_additional_features=True)
        X_ext, y_ext = feature_manager.get_features_target()

        # Set filters_metadata to the provided internal filters_metadata
        config.filters_metadata = filter_metadata_internal

        # Initialize the PredictionManager with the given config
        prediction_manager = PredictionManager(config)

        # Run predictions using the external set
        prediction_manager.get_model_predictions_binaryClassification(X_ext, y_ext, return_predictions=False)
    else:
        # Default behavior without an external set
        prediction_manager = PredictionManager(config)
        prediction_manager.get_model_predictions_binaryClassification(return_predictions=False)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run binary classification predictions.")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--internal_set",
        type=str,
        help="Internal set filters in the form of a JSON string (e.g., '{\"Centre\": \"Vienna\", \"group_test\": \"HCC\"}')."
    )
    args = parser.parse_args()

    # Parse external set filters if provided
    internal_set = None
    if args.internal_set:
        try:
            internal_set = json.loads(args.internal_set)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format for --internal_set: {e}")
            sys.exit(1)

    # Pass the `config_file` argument to the function
    run_binaryClassification(config_file=args.config_file, filter_metadata_internal=internal_set)
