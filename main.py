import numpy as np

def k_means(X, k, max_iters=100):
    indices = np.random.choice(len(X), k, replace=False) 
    centroides = X[indices]

    for _ in range(max_iters):
        clusters = [ [] for _ in range(k) ]

        #em cada ponto da matriz(dataset)
        for ponto in X:
            distancias = np.linalg.norm(ponto - centroides, axis=1)
            cluster_id = np.argmin(distancias)
            clusters[cluster_id].append(ponto)   
        
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


    return centroides, clusters



X = np.array([
    [1,1],
    [2,2],
    [5,5],
    [6,6],
    [8,8],
    [9,9]
])

centroides, clusters = k_means(X, k=3)


print(f"Centroides: {centroides}")
print(f"Clusters: {clusters}")










