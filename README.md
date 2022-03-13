# AzureML - Predicting Heart Attacks

## Project Overview
This project demonstrates how to use Azure ML Studio to train a machine learning model and deploy it to production.

## Dataset
### Overview
For this project a dataset from Kaggle is used. It contains patient health data. It can be found here: https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
### Task
With the provided data it is possible to train a classifier that can predict if a patient has a high risk of a heart attack. The labels are stored in the column ```output```

### Access
To use this data in Azure it needs to be available as a dataset. The easiest way to import the data is via the UI:
![image](https://user-images.githubusercontent.com/56161454/158071445-24ab6cc8-e7d2-4976-ae63-e6a9ebd5771f.png)
After registering the dataset it can be easily retrieved:
```
dataset = Dataset.get_by_name(ws, name='heart')
dataset.to_pandas_dataframe()
```

## Automated ML
The [AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml) feature can be used to train and compare different models with various hyperparameters and find the best one with little effort.
### Config
The most important thing to do is to define the [AutoMLConfig](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py) class.
Basic information include the type of task that needs to be solved, training data, label information, validation methods as well as compute parameters:
```
automl_config = AutoMLConfig(task="classification", 
compute_target=compute_target,
training_data=dataset,
label_column_name="output",
n_cross_validations=5,
experiment_timeout_hours=0.5)
```
Generally speaking the configuration of the AutoML run can be very extensive but you can get away with little as a lot gets figured out automatically.
### Training
After submitting the run together with the AutoMLConfig you can use the [RunDetails](https://docs.microsoft.com/en-us/python/api/azureml-widgets/azureml.widgets.rundetails?view=azure-ml-py) widget to track the progress of the training:
![image](https://user-images.githubusercontent.com/56161454/158079787-56412893-c115-4ad8-9660-f67d58388050.png)

### Results
The best performing model is a VotingEnsemble with an accuracy of **86.5%** containing seven sub models of types logistic regression and random forest:
![image](https://user-images.githubusercontent.com/56161454/158079651-2271fdf9-9051-4f29-863e-0361227e4624.png)
The results could be probably be improved on giving the AutoML run more resources.

## Hyperparameter Tuning
For this example Support Vector Classification ([SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)) is being used.
### Config
The most import class to specify the hyperparameter tuning is the [HyperDriveConfig](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py) class.
Besides the run config, the metric to optimize for and other details of the compute the hyperparameters to tune are specified in a [HyperParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperparametersampling?view=azure-ml-py) class. In this specific example a [BayesianParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.bayesianparametersampling?view=azure-ml-py) was used to speed up the training process and gain better results. This samples tries to pick new values in the parameter space that further improve the primary metric.
For SVC the kernel and the penalty was tuned:
```
ps = BayesianParameterSampling(
        {
            "--penalty": uniform(0.01, 10),
            "--kernel": choice("linear", "poly", "rbf", "sigmoid")
        }
    )
```
### Training
Again the [RunDetails](https://docs.microsoft.com/en-us/python/api/azureml-widgets/azureml.widgets.rundetails?view=azure-ml-py) widget can be used to track the progress of the run:
![image](https://user-images.githubusercontent.com/56161454/158079971-022d42c2-167a-4487-8250-1585c38dd9c3.png)

### Results
The best performing model had an accuracy of **76.3%** with a kernel of type "poly" and a penalty around 7. In this parallel coordinates plot the effect of the hyperparameter on the accuracy can be understood:
![image](https://user-images.githubusercontent.com/56161454/158078864-b52fdfc2-4991-4f16-a1d7-e38be9b8f7a8.png)
It shows that the kernals "linear" and "sigmoid" especially in combination with high penalties perform worse than the kernals "rbf" and "poly".

## Comparision
The accuracy of the VotingEnsemble from the AutoML run is around 10% higher than the accuracy of the best model from the hyperparameter run. This shows that the SVC is not ideal for this problem and AutoML did find a better working model.

## Model Deployment
### Deploying
The best model is deployed to a Azure Container Instance. For the deployment you need the model, a inference config as well as an deployment config. In the inference config the environment as well as the entry script is specified. The deployment config holds information about the physical properties of the machine that is used for excecution. It is usefull to enable application insights for the webservice. In application insights the logs as well as metrics of the service can be monitored to analyse the performance or any errors of the model:
```
inference_config = InferenceConfig(
    environment=env,
    source_directory="./source_dir_automl",
    entry_script="./scoring_file_v_1_0_0.py",
)
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores = 1,
    memory_gb = 1,
    auth_enabled=True,
    enable_app_insights=True)
service = Model.deploy(
    ws,
    "heart-predict",
    [best_model],
    inference_config,
    deployment_config,
    overwrite=True,
)
```

### Query the Model Endpoint with the SDK
There are multiple ways to query the model endpoint. The easiest is to use the [Webservice](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice(class)?view=azure-ml-py) class from azureml SDK. It is as easy as:
```
import json
data = {
  "data": [
    {"age": 20, "sex": 1, "cp": 0, "trtbps": 0, "chol": 0, "fbs": 0, "restecg": 0, "thalachh": 0, "exng": 0, "oldpeak": 0, "slp": 0, "caa": 0, "thall": 0
    }
  ],
  "method": "predict"
}
input_data = json.dumps(data)
response = service.run(input_data)
print(response)
```

### Query the Model Endpoint via HTTP
The universal way to query the endpoint is by sending a HTTP request. The scoring URI as well as the key can be retrieved via the UI or the SDK:
```
import requests

scoring_uri = service.scoring_uri
key = service.get_keys()[0]
headers = {"Content-Type": "application/json",
"Authorization": f"Bearer {key}"}
response = requests.post(scoring_uri, input_data, headers=headers)
print(response)
print(response.json())
```
![image](https://user-images.githubusercontent.com/56161454/158080276-77a32d59-fd8d-4015-90c8-dd0a3afb3350.png)


## Screen Recording
In this [screencast](https://youtu.be/qqRtPvKxeqA) the deployed model is shown in action.
