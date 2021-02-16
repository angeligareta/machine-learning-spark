<h1 align="center">Machine Learning Overview with Spark ML</h1>
<h4 align="center">First lab of the Scalable Machine Learning course of the EIT Digital data science master at <a href="https://www.kth.se/en">KTH</a></h4>

<p align="center">
  <img alt="KTH" src="https://img.shields.io/badge/EIT%20Digital-KTH-%231954a6?style=flat-square" />  
  <img alt="License" src="https://img.shields.io/github/license/angeligareta/machine-learning-spark?style=flat-square" />
  <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/angeligareta/machine-learning-spark?style=flat-square" />
</p>

## Problem Statement
This project aims to study the basics of regression and classification in Spark. It is divided in two parts:
- **First part:** guided exercise whose objective is to predict median housing value in the dataset [California Housing Data (1990)](https://www.kaggle.com/harrywang/housing), 
which involves the analysis and transformation of the attributes of the dataset (e.g., one-hot encoding, string indexer, normalization...).. After that, four different regression models
are implemented: linear regression, decision tree, random forest and gradient-boost forest regression. Finally, the dataset is divided in train and test sets, and the models are trained and hypertuned.
- **Second part:** aims to classify the default payment for credit card customers , by using the dataset [Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset).
First, a explanatory analysis will be performed over the data, followed by the implementation and training of three different classification models (logistic regression, decision tree, and random forest). Finally,
the models would be compared and a brief discussion about which model performs better for the task.

## Tools
The implementation of both parts of the assignments is performed using Scala programming language with Apache Spark Machine Learning library. In addition, [Databricks](https://community.cloud.databricks.com/) was used
to train more efficiently in a cluster, so the source format consists of a Scala notebook. The implementation can be found at [src/](src/) and the notebook preview at [https://angeligareta.com/machine-learning-spark/](https://angeligareta.com/machine-learning-spark/). 

## Authors
- Serghei Socolovschi [serghei@kth.se](mailto:serghei@kth.se)
- Angel Igareta [angel@igareta.com](mailto:angel@igareta.com)
