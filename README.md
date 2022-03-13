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

## Automated ML
The [AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml) feature can be used to train and compare different models with various hyperparameters with little effort.
The most important thing to do is to define the [AutoMLConfig](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py) class.
