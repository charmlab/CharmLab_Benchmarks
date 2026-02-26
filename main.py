# generic example of a full end to end run of the repo
from data_layer.data_module import DataModule
from evaluation_layer.distances import Distance
from evaluation_layer.evaluation_module import EvaluationModule
from method_layer.ROAR.method import ROAR
from model_layer.model_module import ModelModule
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Step 1: Initialize the DataModule with the path to the data config YAML
    data_module = DataModule(config_path="data_config_adult.yml")
    
    # Step 2: Initialize the ModelModule with the path to the model config YAML and the processed DataModule
    model_module = ModelModule(config_path="model_config_mlp.yml", data_module=data_module)
    
    # Step 3: Initialize the method module with the DataModule and ModelModule
    method = ROAR(data_module, model_module) 
    
    # Step 4: Make predictions on new data (example input)
    X_test, y_test = model_module.get_test_data()
    predictions = model_module.predict(X_test)
    negative_class_indices = np.where(predictions == 0)[0]

    factuals = pd.DataFrame(X_test[negative_class_indices][:5], columns=data_module.get_feature_names(expanded=True))

    # now generate counterfactuals for these factuals using ROAR
    counterfactuals = method.get_counterfactuals(factuals)

    # perform some benchmarking of the method using the evaluation module
    evaluation_module = Distance(data_module)

    evaluation_results = evaluation_module.get_evaluation(factuals, counterfactuals)
    print(evaluation_results)