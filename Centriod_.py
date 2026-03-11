from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2],
              [1, 4],
              [5, 8],
              [6, 9]])

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

print("Centroids:")
print(kmeans.cluster_centers_)

print("Labels:", kmeans.labels_)
