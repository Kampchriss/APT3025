# APT3025
Clustering
For K_means.ipynb
# Movie Recommendation Clustering

This project demonstrates movie recommendation through clustering users based on their movie ratings using K-Means clustering. The code is implemented in Python using Google Colab.

## Overview

The repository contains Python code that performs the following tasks:

- Loading and exploring user ratings for different movies.
- Applying K-Means clustering to group users into two clusters based on their movie ratings.
- Predicting cluster labels for new users using test data.

## Code Structure

- `clustering_movie_recommendation.ipynb`: Jupyter Notebook containing the code for the movie recommendation clustering.

## Getting Started

1. Open the `clustering_movie_recommendation.ipynb` notebook in Google Colab.
2. Execute the cells in order to load the data, perform clustering, and make predictions.
3. Explore the results and predictions.

## Dependencies

- pandas
- scikit-learn

Install dependencies using:

```bash
pip install pandas scikit-learn

Test Data
The test data (testData) contains ratings for new users to predict their cluster labels.


FOR THE NEW_K_MEANS.IPYNB

# Cluster Analysis Project

## Overview

This project focuses on performing cluster analysis using the K-Means clustering algorithm. The dataset, "cluster_analysis.csv," is utilized to identify patterns and groupings within the given features.

## Getting Started

### Prerequisites

Ensure you have the required dependencies installed. You can install them using:

```bash
pip install pandas scikit-learn numpy
Installation
Clone the repository and navigate to the project directory:

bash
Copy code
git clone <repository-url>
cd <repository-directory>
Usage
Data Exploration
Import the necessary libraries:
python
Copy code
import pandas as pd
from sklearn import cluster
import numpy as np
Load the dataset:
python
Copy code
ratings = pd.read_csv("cluster_analysis.csv")
Create a DataFrame from the selected columns:
python
Copy code
titles = ['Feature_1', 'Feature_2']
movies = pd.DataFrame(ratings, columns=titles)
Explore the dataset:
python
Copy code
print(movies.head())
K-Means Clustering
Perform K-Means clustering:
python
Copy code
k_means = cluster.KMeans(n_clusters=2, max_iter=100, random_state=1)
k_means.fit(movies)
labels = k_means.labels_
Create a DataFrame with cluster assignments:
python
Copy code
outcome = pd.DataFrame({'Feature_1': movies['Feature_1'], 'Feature_2': movies['Feature_2'], 'Cluster ID': labels})
print(outcome.head())
Access cluster centers:
python
Copy code
print(k_means.cluster_centers_)
Prediction on New Data
Read new data for prediction:
python
Copy code
cluster_analysis = pd.read_csv("cluster_analysis.csv")
Create an array for prediction:
python
Copy code
feature_1_values = cluster_analysis['Feature_1'].values
feature_2_values = cluster_analysis['Feature_2'].values
testData = np.column_stack((feature_1_values, feature_2_values))
Predict cluster labels for new data:
python
Copy code
labels_test = k_means.predict(testData)
print(labels_test)


