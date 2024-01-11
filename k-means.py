import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")
features = data[['age', 'bmi']]
k = int(input("Masukan Jumlah Cluster : "))
kmeans = KMeans(n_clusters=k)
data['Cluster_KMeans'] = kmeans.fit_predict(features)

plt.scatter(data['age'], data['bmi'], c=data['Cluster_KMeans'], cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='black', label='Centroids')
plt.xlabel('age')
plt.ylabel('bmi')
plt.title(f'Cluster_KMeans Clustering')
plt.legend()
plt.show()



