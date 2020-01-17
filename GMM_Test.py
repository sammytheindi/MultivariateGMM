from GMM import GMM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=1000, centers=3, n_features=2)

gmm_cls = GMM()
gmm_cls.fit(X, 4)

colors = []
for l in gmm_cls.kmeans_cls_.memberships_:
    if l == 0:
        colors.append('red')
    if l == 1:
        colors.append('green')
    if l == 2:
        colors.append('orange')
    if l == 3:
        colors.append('blue')

plt.scatter(X[:,0], X[:,1], c=colors, alpha=0.1)
plt.scatter(gmm_cls.means_[:, 0], gmm_cls.means_[:, 1], c='k')
plt.show()

plt.scatter(X[:,0], X[:,1], c=colors, alpha=0.2)
plt.scatter(gmm_cls.kmeans_cls_.means_[:, 0], gmm_cls.kmeans_cls_.means_[:, 1], c='k')
plt.show()
