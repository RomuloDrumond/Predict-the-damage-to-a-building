# Overview
This repository contains a quick attempt to build a good predictive model for this competition:  https://www.hackerearth.com/problem/machine-learning/predict-the-energy-used-612632a9-3f496e7f/ 

On the link above you may find the dataset as it was too large for uploading to this repository.

The analysis was split into many jupyter notebooks, as you may find below. 
To install dependencies run `pip install -r requirements.txt` on the main directory.

## [01 Analysis and preprocessing of the data](https://nbviewer.jupyter.org/github/RomuloDrumond/Predict-the-damage-to-a-building/blob/master/01%20Analysis%20and%20preprocessing%20of%20the%20data.ipynb)

There you can find me:

* Merging datasets;
* Dealing with *NaN*'s;
* Converting categorical features to numeric;
* Leaving the dataset ready for model fitting.

### Important libraries used:

* Pandas
* Numpy

## [02 Building and evaluation of ML models](https://nbviewer.jupyter.org/github/RomuloDrumond/Predict-the-damage-to-a-building/blob/master/02%20Building%20and%20evaluation%20of%20ML%20models.ipynb)

Methodology used:

* For each model was run 10 resamplings, or 10 train/test splits with fitting and evaluation of the model;
* The performance metric adopted in the competition was the f1-score so it was used here too. As our evaluation metric is a random variable due to the resampling, it has a mean and standard deviation, making necessary the creation of an objective function with the statistics of the random variable:

<p align="center">
  <img width="250" src="https://latex.codecogs.com/svg.latex?%5Ctext%7Bmaximize%7D%20%5Cquad%20f_o%28u%29%20%3D%20%5Cmu%20%28u%29%20-%202%5Csigma%28u%29">
</p>

We can say that the objective function tries to balance the two objectives: higher mean, or better generalization, with a lower standard deviation, or lower instability.

### Important libraries used:

* Sklearn
* XGBoost
* Keras (Tensorflow-gpu backend)

### Algorithms used:

1. k-Nearest Neighbors (k-NN);
2. Linear Regression;
3. Logistic Regression;
4. Nearest Centroid Classifier (NCC);
5. Quadratic Gaussian Classifier (QGC);
6. Decision Trees;
7. Artificial Neural Networks.


## [03 Dimensionality reduction and reevaluation of models](https://nbviewer.jupyter.org/github/RomuloDrumond/Predict-the-damage-to-a-building/blob/master/03%20Dimensionality%20reduction%20and%20reevaluation%20of%20models.ipynb)

In this notebook we explore the effects of dimensionality reduction, using **Principal Component Analysis (PCA)**, on the performance and training time of the classifiers used.

As a hyperparameter, the conserved variance, p, was set to 98%. Also, the type of feature scaling was set to min-max as many transformed categorical features now have values 0 or 1, so, using standard scaling may result in the original numerical features dominating the PCA transformation, as they would have values ranging from -1 to +1.


## [04 Choosing the best model and producing the submission file](https://nbviewer.jupyter.org/github/RomuloDrumond/Predict-the-damage-to-a-building/blob/master/04%20Choosing%20the%20best%20model%20and%20producing%20the%20submission%20file.ipynb)

The tree-based model was adopted as it had shown better generalization and stability. A final model was fit in whole train data set, predictions were made in the test data set and saved in the `submission.csv` file.
