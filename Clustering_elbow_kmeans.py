# minimal KMeans + plot (no defs)
import numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([[4,21],[5,19],[10,24],[4,17],[3,16],[11,25],[14,24],[6,22],[10,21],[12,21]])
km = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=0).fit(X)

plt.scatter(X[:,0], X[:,1], c=km.labels_)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], marker='X', s=100, edgecolor='k')
plt.show()


# tiny elbow (inertia) check for k=1..5
from sklearn.cluster import KMeans
inertia = [KMeans(n_clusters=i, random_state=0).fit(X).inertia_ for i in range(1,6)]
import matplotlib.pyplot as plt
plt.plot(range(1,6), inertia, 'o-'); plt.xlabel('k'); plt.ylabel('inertia'); plt.show()


import pandas as pd, numpy as np
from sklearn.cluster import KMeans

np.random.seed(42)

data = {'AnnualIncome': np.random.randint(30000, 100000, 100),
        'SpendingScore': np.random.randint(1, 100, 100)}
df = pd.DataFrame(data)

X = df.values
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
df['Cluster'] = kmeans.labels_

# fixed input point
new_point = np.array([[45000, 60]])   # [AnnualIncome, SpendingScore]
print("Predicted cluster for", new_point[0], "is:", kmeans.predict(new_point)[0])
