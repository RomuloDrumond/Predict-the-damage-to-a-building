# Overview
This repository contains a quick attempt to build a good predictive model for this competition:  https://www.hackerearth.com/problem/machine-learning/predict-the-energy-used-612632a9-3f496e7f/ 

On the link above you may find the dataset as it was too large for uploading to this repository.

The analysis was split into many jupyter notebooks, as you may find below. 
To install dependencies run `pip install -r requirements.txt` on the main directory.

## [01 Analyzing and preprocessing data](https://github.com/RomuloDrumond/Predict-the-damage-to-a-building/blob/master/01%20Analyzing%20and%20preprocessing%20data.ipynb)

There you can find me:

* Merging datasets;
* Dealing with *NaN*'s;
* Transforming categorical features to numeric;
* Leaving the dataset ready for model fitting.

### Important libraries used:

* Pandas
* Numpy

## [02 Models building and evaluation](https://github.com/RomuloDrumond/Predict-the-damage-to-a-building/blob/master/02%20Models%20building%20and%20evaluation.ipynb)

Methodology used:

* For each model was run 10 resamplings, 10 train/test splits with fitting and evaluation of the model;
* The performance metric adopted for evaluating models in the competition was the f1-score so it was used here too. As our performance metric is a random variable due to the resampling, it has a mean and standard deviation, making necessary the creation of an objective function with the statistics of the random variable:

<p align="center">
  <img width="250" src="http://www.sciweavers.org/download/Tex2Img_1561408865.jpg">
</p>

We can say that the objective function tries to balance the two objectives: higher mean, or better generalization, with a lower standard deviation, or lower instability.

### Important libraries used:

* Sklearn
* Xgboost
* Keras (Tensorflow-gpu backend)

### Algorithms used:

1. k-Nearest Neighboors (k-NN)
2. Linear Regression
3. Logistic Regression
4. Nearest Centroid Classifier (NCC)
5. Quadratic Gaussian Classifier (QGC)
6. Decision Trees
7. Artificial Neural Networks
