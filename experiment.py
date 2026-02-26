# generic example of a full end to end run of the repo
from data_layer.data_module import DataModule
from evaluation_layer.distances import Distance
from model_layer.model_module import ModelModule
from method_layer.ROAR.method import ROAR
import numpy as np
import pandas as pd

if __name__ == "__main__":

    data_module = DataModule(
        data_path="data_layer/raw_csv/german.csv",
        config_path="data_layer/config_files/data_config_german.yml")
    
    print("here is the processed data:")
    print(data_module.get_processed_data().head())

    model_module = ModelModule(
        config_path="model_layer/model_config_mlp.yml",
        data_module=data_module
    )

    # get model accuracy
    train_accuracy = model_module.get_train_accuracy()
    print(f"Model training accuracy: {train_accuracy}")
    accuracy = model_module.get_test_accuracy()
    print(f"Model test accuracy: {accuracy}")

    # test to see if ROAR method runs without error
    method = ROAR(data_module, model_module)

    # get some factuals to generate counterfactuals for
    X_test, y_test = model_module.get_test_data()

    # get the first 5 rows of the processed test data as factuals
    # specifically, we can the ones predicted as the negative class (label 0) 
    predictions = model_module.predict(X_test)
    negative_class_indices = np.where(predictions == 0)[0]

    factuals = pd.DataFrame(X_test[negative_class_indices][:5], columns=data_module.get_feature_names(expanded=True))

    print("Here are the factuals we will generate counterfactuals for:")
    print(factuals)

    # now generate counterfactuals for these factuals using ROAR
    counterfactuals = method.get_counterfactuals(factuals)
    print("Here are the generated counterfactuals:")
    print(counterfactuals)

    # perform some benchmarking of the method using the evaluation module
    evaluation_module = Distance(data_module)

    evaluation_results = evaluation_module.get_evaluation(factuals, counterfactuals)
    print("Here are the evaluation results for the generated counterfactuals:")
    print(evaluation_results)