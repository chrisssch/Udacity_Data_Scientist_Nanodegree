# Course Projects for Udacity's Data Scientist Nanodegree - Term 1

This repository contains (updated versions of) my course projects for the first term of [Udacity's Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025). This term includes three courses:

1. Supervised Learning
2. Unsupervised Learning
3. Deep Learning

A short description of the three course projects:


## Supervised Learning: Finding Donors for CharityML

*Originally uploaded in November 2018*

In this project, the annual income bracket of people based on demographic data is predicted using supervised machine learning models. The following tasks are performed:
* Exploratory analysis of the dataset
* Preprocessing the data for use in machine learning
* Training three classification models with default hyperparameters; I chose logistic regression, random forest and support vector machine models
* Tuning the best performing model
* Selecting the most important features


## Unsupervised Learning: Identifying Customer Segments

*Originally uploaded in November 2018*

The goal of this project is to apply unsupervised learning techniques to identify segments of the population that form the core customer base for a company. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. This project involved exploratory analysis, data preprocessing, principal component analysis to reduce the dimensionality of the dataset, clustering, and comparative analysis.

The data used in this project is real-life data provided by a partner of Udacity. As this data is confidential, I had to anonymize it. Among other things, all references to the company and all column names were removed. Without this context, the content of the data is meaningless.


## Deep Learning: Image Classifier Application

*Originally uploaded in December 2018*

The goal of this project is to build a command line application in Python and PyTorch that predicts the species of a flower based on an image. I use pre-trained convolutional neural network, add a few layers and train them on a dataset of images of flowers of about 100 species. After training for an hour or so on my GPU, the model achieves an accuracy of around 70% on the test set.
The first part of this project is to develop an initial prototype in a Jupyter notebook. In the second part, the code is converted into functions in Python scripts for training and predicting that take number of arguments such as selecting a pre-trained model, the device for training, hyperparameters, and so on.
