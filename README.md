
# README: Electrical Grid Stability Simulated Data Analysis

## Table of Contents
1. [Introduction](#1-introduction)
2. [Dataset Overview](#2-dataset-overview)
   - 2.1. [Source](#21-source)
   - 2.2. [Features](#22-features)
   - 2.3. [Target Variable](#23-target-variable)
3. [Project Structure](#3-project-structure)
4. [Requirements](#4-requirements)
5. [Data Loading and Preprocessing](#5-data-loading-and-preprocessing)
   - 5.1. [Data Splitting](#51-data-splitting)
6. [Modeling](#6-modeling)
   - 6.1. [Support Vector Machine (SVM)](#61-support-vector-machine-svm)
   - 6.2. [K-Nearest Neighbors (KNN)](#62-k-nearest-neighbors-knn)
   - 6.3. [Decision Tree](#63-decision-tree)
7. [Performance Evaluation](#7-performance-evaluation)
   - 7.1. [Metrics Used](#71-metrics-used)
   - 7.2. [Optimal Hyperparameters](#72-optimal-hyperparameters)
8. [Hyperparameter Tuning in Machine Learning](#8-hyperparameter-tuning-in-machine-learning)
9. [Results](#9-results)
10. [References](#10-references)

---

## 1. Introduction
This project analyzes the **Electrical Grid Stability Simulated Data** from the UCI Machine Learning Repository to classify grid stability (target variable: `stabf`). The classification is achieved using various machine learning models like SVM, KNN, and Decision Trees. The performance of each model is evaluated based on accuracy, and optimal hyperparameters are determined.

## 2. Dataset Overview

### 2.1. Source
The dataset is publicly available on the UCI Machine Learning Repository:
- [Electrical Grid Stability Simulated Data](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data)

### 2.2. Features
The dataset contains 12 input features related to grid stability:
1. `tau1` - Time constant (source)
2. `tau2` - Time constant (sink)
3. `p1` - Power input
4. `p2` - Power output
5. `g1` - Conductance
6. `g2` - Conductance
7. `K1` - Input coefficient 1
8. `K2` - Input coefficient 2
9. `K3` - Output coefficient 1
10. `K4` - Output coefficient 2
11. `u1` - Control variable
12. `u2` - Control variable

### 2.3. Target Variable
- `stabf` - Binary classification of grid stability (`stable` or `unstable`).

## 3. Project Structure
- `JakeTolentino_GridStability.ipynb`: Jupyter Notebook containing data loading, preprocessing, model implementation, and evaluation.
- `data/`: Directory where the dataset is stored.
- `README.md`: Project documentation.

## 4. Requirements
To run this project, the following Python libraries are required:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

## 5. Data Loading and Preprocessing

### 5.1. Data Splitting
The dataset is split into training and test sets using an 80/20 split. The same split is used across all models (SVM, KNN, Decision Tree).

## 6. Modeling

### 6.1. Support Vector Machine (SVM)
Three different kernels are used for classification:
- Linear Kernel
- Polynomial Kernel
- Radial Basis Function (RBF) Kernel

Performance is evaluated on each kernel, and hyperparameters like `C` and `gamma` are tuned using grid search.

### 6.2. K-Nearest Neighbors (KNN)
The K-Nearest Neighbors model is implemented, and the number of neighbors (`K`) is tuned on the training set. The optimal `K` is selected based on performance on the test set.

### 6.3. Decision Tree
A Decision Tree classifier is used, and the maximum depth of the tree is tuned on the training set. The optimal depth is chosen based on test set performance.

## 7. Performance Evaluation

### 7.1. Metrics Used
Performance of each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

### 7.2. Optimal Hyperparameters
The optimal hyperparameters for each model are reported after grid search cross-validation.

## 8. Hyperparameter Tuning in Machine Learning
This section provides a summary of the hyperparameters used across the models (SVM, KNN, Decision Tree) and discusses the importance of tuning for each model to improve performance.

## 9. Results
The results section includes the performance of each model (SVM, KNN, Decision Tree) on the test set, along with a comparison of accuracy and other evaluation metrics.

## 10. References
- UCI Machine Learning Repository: [Electrical Grid Stability Simulated Data](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data)
- IEEE Referencing Guide: [IEEE Style Guide](https://www.bath.ac.uk/publications/library-guides-to-citing-referencing/attachments/ieee-style-guide.pdf)
