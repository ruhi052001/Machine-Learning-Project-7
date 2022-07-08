# -*- coding: utf-8 -*-
"""


@author: Priyanka Kumari
      roll no : B20307
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.cluster import DBSCAN
import scipy as sp
from scipy import spatial as spatial
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn import decomposition


# func for purity score from snippets
def purity_score(y, pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y, pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


# loading data
df = pd.read_csv("Iris.csv")
df1 = df.drop(columns=['Species'])

# Assigning class labels
actual_label = []
for i in df['Species']:
    if i == 'Iris-setosa':
        actual_label.append(0)
    elif i == 'Iris-virginica':
        actual_label.append(2)
    else:
        actual_label.append(1)

print("-----------------------------------------------------Q1--------------------------------------------------------")

# using pca analysis to convert 4d data to 2d data
pca = decomposition.PCA(n_components=2)
df2 = pca.fit_transform(df1)
df2 = pd.DataFrame(df2, columns=['A', 'B'])
print(round(df2.cov(), 3))
print(pca.explained_variance_ratio_)
# scatter plot of dimensionally reduced data
plt.scatter(df2['A'], df2['B'])
plt.xlabel("A")
plt.ylabel("B")
plt.title("Plot of data after dimensionality reduction:")
plt.show()

# plot of eigen values vs components
val, vec = np.linalg.eig(df.corr().to_numpy())
t = np.linspace(1, 4, 4)
plt.plot(t, [round(i, 3) for i in val])
plt.xticks(np.arange(min(t), max(t) + 1, 1.0))
plt.xlabel("Components")
plt.ylabel("Eigen Values")
plt.title("Eigen Values Vs Components")
plt.show()

print("-----------------------------------------------------Q2--------------------------------------------------------")
# k means clustering
K = 3
kmeans = KMeans(n_clusters=K)
kmeans.fit(df2)
kmeans_prediction = kmeans.predict(df2)

# scatter plot of different clusters obtained from k means clustering
plt.scatter(df2['A'], df2['B'], c=kmeans_prediction, cmap='rainbow', s=13)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title("K-mean Clustering")
plt.show()

# distortion measure
print("Distortion measure: ", round(kmeans.inertia_), 3)

# purity score
print('Purity Score =', round(purity_score(actual_label, kmeans_prediction), 3))

print("-----------------------------------------------------Q3--------------------------------------------------------")

K = [2, 3, 4, 5, 6, 7]

distortion = []
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df2)
    kmeans_prediction = kmeans.predict(df2)
    distortion.append(kmeans.inertia_)
    print(f"Purity Score {k}: ", round(purity_score(actual_label, kmeans_prediction), 3))

plt.plot(K, distortion)
plt.title("Elbow Method on K-Means Clustering")
plt.xticks(K)
plt.xlabel("Values of K")
plt.ylabel("Distortion")
plt.show()

print("---------------------------------------------------Q4----------------------------------------------------")
# gmm based clustering
K = 3
gmm = GaussianMixture(n_components=K)
gmm.fit(df2)
GMM_prediction = gmm.predict(df2)
# scatter plot of different clusters obtained using gmm clustering
plt.scatter(df2["A"], df2["B"], c=GMM_prediction, cmap='rainbow', s=13)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='black')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title("GMM based clustering")
plt.show()
# gmm distortion measure
print("Total data log likelihood: ", round(gmm.score(df2) * len(df2)), 3)
# purity score
print('Purity Score =', purity_score(actual_label, GMM_prediction))

print("--------------------------------------------------Q5-----------------------------------------------------------")
# GMM as soft-clustering techniques
K = [2, 3, 4, 5, 6, 7]
distortion = []
for k in K:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(df2)
    GMM_prediction = gmm.predict(df2)
    distortion.append(gmm.score(df2) * len(df2))
    purity_score(actual_label, GMM_prediction)
    print(f"Purity Score {k}: ", round(purity_score(actual_label, GMM_prediction), 3))

# plot of K vs total data log likelihood
plt.plot(K, distortion)
plt.title("K vs Total data log likelihood")
plt.xticks()
plt.xlabel("Values of K")
plt.ylabel("Distortion")
plt.show()

print("-----------------------------------------------------Q6--------------------------------------------------------")

# DBSCAN clustering using different values for eps and min_samples
eps = [1, 1, 5, 5]
min_samples = [4, 10, 4, 10]
print(" ")
for i in range(4):
    dbscan_model = DBSCAN(eps=eps[i], min_samples=min_samples[i]).fit(df2)
    DBSCAN_predictions = dbscan_model.labels_
    # data of different clusters
    label_0_dbscan = df2[DBSCAN_predictions == 0]
    label_1_dbscan = df2[DBSCAN_predictions == 1]

    # scatter plot of different clusters by DBSCAN
    plt.scatter(label_0_dbscan['A'], label_0_dbscan['B'], color='purple')
    plt.scatter(label_1_dbscan['A'], label_1_dbscan['B'], color='green')
    plt.xlabel("Dimension A")
    plt.ylabel("Dimension B")
    plt.title(f"Clustering using DBSCAN for eps={eps[i]} and min_sample={min_samples[i]}")
    plt.show()

    # purity score
    p = round(purity_score(actual_label, DBSCAN_predictions), 3)
    print(f"Purity Score for eps={eps[i]}, min_samples={min_samples[i]} is {p}")