import logging
import sys
from dataHandler import Config
from modelEvaluator import PredictionManager

DATA_DIR ='/gpfs/data/fs71974/creynablanco/Projects/HCC/Data'
METADATA_DIR= '/gpfs/data/fs71974/creynablanco/Projects/HCC/Metadata'
LIB_META_DATA = 'aligent_twist_with_important_info_excludeSARS.pkl'
PROJECT='HCC_MUW'
group_tests = ['Controls', 'HCC']

COL_TARGET = 'group_test'
EXTRA_FEATURES_TO_INCLUDE = ['Sex', 'Age']
filters_metadata = {'group_test': ['Controls', 'HCC']}

def train_test_binary_predictions():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        stream=sys.stdout
    )

    # Initialize the configuration.
    # You might load from a YAML/JSON file or set defaults directly.
    config = Config(
        data_dir=DATA_DIR,
        metadata_dir=METADATA_DIR,
        lib_meta_data=LIB_META_DATA,
        project=PROJECT,
        group_tests=group_tests,
        col_target=COL_TARGET,
        extra_features_to_include=EXTRA_FEATURES_TO_INCLUDE,
        filters_metadata=filters_metadata,
        k = 10, tuning_k = 5, tuning_n_iter = 30, with_run_plates_options=[False]
                , filter_by_entropy=[False], with_oligos_options=[True, False], with_additional_features_options=[False, True], prevalence_thresholds_min=[2, 5, 10, 20, 50],
                cv_method = 'kfold', split_train_test = True, train_size = 0.8, external_set = False, tuning_parameters=True,
                compute_feature_importance= False, return_train = True, return_test = True
    )

    # Initialize the PredictionManager with the given config
    prediction_manager = PredictionManager(config)


    # Run nested cross-validation and summarize the models
    # This will generate prediction CSVs and a summary pickle
    prediction_manager.get_model_predictions_binaryClassification(return_predictions=False)


if __name__ == "__main__":
    train_test_binary_predictions()
