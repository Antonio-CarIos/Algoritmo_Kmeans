import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt


X = np.array([
    [1,1],
    [1.5,2],
    [3,4],
    [5,7],
    [3,5],
    [4,5],
    [8,8],
    [9,9],
    [3,6],
    [2,2],
    [9,2],
    [9,3]
    
])


def k_means(X, k, max_iters=600):

    indices = np.random.choice(len(X), k, replace=False)

    centroides = X[indices]


    for _ in range(max_iters):

        clusters = [[] for _ in range(k)]

        labels = np.zeros(len(X), dtype=int)


        for idx, ponto in enumerate(X):

            distancias = np.linalg.norm(ponto - centroides, axis=1)

            cluster_id = np.argmin(distancias)

            clusters[cluster_id].append(ponto)

            labels[idx] = cluster_id


        novos_centroides = []

        for i, cluster in enumerate(clusters):

            if len(cluster) > 0:
                novo_centroide = np.mean(cluster, axis=0)

                novos_centroides.append(novo_centroide)

            else:
                novos_centroides.append(centroides[i])

    
        novos_centroides = np.array(novos_centroides)


        if np.allclose(centroides, novos_centroides):
            break

        centroides = novos_centroides

    return centroides, clusters, labels



centroides, clusters, labels = k_means(X, k=4)


print(f"Centroides:\n{centroides}\n")

print("Clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i}: {cluster}")

print(f"\nLabels: {labels}")


for i in range(len(X)):

    plt.scatter(
        X[i][0],               
        X[i][1],               
        c=f"C{labels[i]}",     
        s=100                  
    )


plt.scatter(
    centroides[:, 0],          
    centroides[:, 1],          
    c="black",                 
    marker="X",                
    s=300,                     
    label="Centroides"
)

plt.title("K-Means")
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")

plt.grid(True)

plt.legend()

plt.show()