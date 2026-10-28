import numpy as np


def _kmeans_plus_plus(X, k):
    """
    Escolhe os centroides iniciais utilizando o algoritmo K-Means++.

    X: dataset com formato (n_amostras, n_caracteristicas)
    k: quantidade de clusters
    """

    # Quantidade de pontos existentes no dataset
    n_amostras = len(X)

    # Escolhe aleatoriamente o primeiro centroide
    primeiro_indice = np.random.randint(n_amostras)

    centroides = [X[primeiro_indice]]

    # Continua escolhendo até obter k centroides
    for _ in range(1, k):

        # Calcula a distância de cada ponto para o
        # centroide mais próximo já escolhido
        distancias = np.array([
            np.min(
                np.linalg.norm(
                    ponto - np.array(centroides),
                    axis=1
                )
            )
            for ponto in X
        ])

        # O K-Means++ utiliza a distância ao quadrado
        # para definir a probabilidade de escolha
        distancias_quadradas = distancias ** 2

        soma = np.sum(distancias_quadradas)

        # Caso todos os pontos já tenham distância zero,
        # escolhe um ponto aleatório ainda não utilizado
        if soma == 0:

            indices_disponiveis = [
                i for i in range(n_amostras)
                if not any(
                    np.array_equal(X[i], centroide)
                    for centroide in centroides
                )
            ]

            novo_indice = np.random.choice(
                indices_disponiveis
            )

        else:

            # Transforma as distâncias em probabilidades
            probabilidades = (
                distancias_quadradas / soma
            )

            # Escolhe um novo centroide de acordo
            # com essas probabilidades
            novo_indice = np.random.choice(
                n_amostras,
                p=probabilidades
            )

        centroides.append(X[novo_indice])

    return np.array(centroides)


def _k_means_once(X, k):

    # Inicialização utilizando K-Means++
    centroides = _kmeans_plus_plus(X, k)

    while True:

        clusters = [[] for _ in range(k)]

        labels = np.zeros(
            len(X),
            dtype=int
        )

        # =====================================
        # ATRIBUIÇÃO DOS PONTOS
        # =====================================

        for idx, ponto in enumerate(X):

            # Calcula a distância do ponto para
            # todos os centroides
            distancias = np.linalg.norm(
                ponto - centroides,
                axis=1
            )

            # Escolhe o centroide mais próximo
            cluster_id = np.argmin(distancias)

            clusters[cluster_id].append(ponto)

            labels[idx] = cluster_id

        # =====================================
        # RECÁLCULO DOS CENTROIDES
        # =====================================

        novos_centroides = []

        for i, cluster in enumerate(clusters):

            if len(cluster) > 0:

                # A média é calculada por característica.
                #
                # Exemplo para 3 dimensões:
                #
                # [x1, y1, z1]
                # [x2, y2, z2]
                #
                # resultado:
                #
                # [media_x, media_y, media_z]

                novo_centroide = np.mean(
                    cluster,
                    axis=0
                )

                novos_centroides.append(
                    novo_centroide
                )

            else:

                # Mantém o centroide anterior
                # caso o cluster fique vazio
                novos_centroides.append(
                    centroides[i]
                )

        novos_centroides = np.array(
            novos_centroides
        )

        # =====================================
        # CRITÉRIO DE PARADA
        # =====================================

        # Para quando os centroides não mudarem
        if np.allclose(
            centroides,
            novos_centroides
        ):
            break

        centroides = novos_centroides

    return (
        centroides,
        clusters,
        labels
    )


def calcular_inercia(
    centroides,
    clusters
):

    inercia = 0

    for i, cluster in enumerate(clusters):

        centroide = centroides[i]

        for ponto in cluster:

            # Funciona para qualquer quantidade
            # de dimensões.
            #
            # 2D:
            # (x1-x2)^2 + (y1-y2)^2
            #
            # 3D:
            # (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2
            #
            # nD:
            # soma das diferenças ao quadrado
            # em todas as características.

            inercia += np.sum(
                (ponto - centroide) ** 2
            )

    return inercia


def calcular_elbow(
    X,
    k_min=2,
    k_max=10,
    n_init=10
):
    """
    Executa o K-Means para cada valor de k na faixa
    [k_min, k_max] e retorna as inércias correspondentes.
    Utilizado no Elbow Method para visualizar o k ideal.
    """

    ks = list(range(k_min, k_max + 1))
    inercias = []

    for k in ks:

        centroides, clusters, _ = k_means(
            X,
            k,
            n_init=n_init
        )

        inercias.append(
            calcular_inercia(
                centroides,
                clusters
            )
        )

    return ks, inercias


def _calcular_silhueta_com_distancias(
    distancias,
    labels
):
    """
    Calcula o coeficiente de silhueta médio a partir
    da matriz de distâncias e dos labels do cluster.

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

    - a(i): distância média ao mesmo cluster
    - b(i): menor distância média a outro cluster
    """
    n = len(labels)

    if n <= 2:
        return 0.0

    rotulos = np.unique(labels)

    if len(rotulos) < 2:
        return 0.0

    soma = 0.0

    for i in range(n):

        mesmo_cluster = labels == labels[i]
        tamanho = np.sum(mesmo_cluster)

        # Cluster com 1 ponto não possui silhueta válida
        if tamanho <= 1:
            continue

        # Média das distâncias ao mesmo cluster
        # (o próprio ponto tem distância 0)
        a = np.sum(distancias[i][mesmo_cluster]) / (tamanho - 1)

        # Menor média das distâncias a outro cluster
        b = np.inf

        for rotulo in rotulos:

            if rotulo == labels[i]:
                continue

            outro_cluster = labels == rotulo

            if np.any(outro_cluster):
                b = min(
                    b,
                    np.mean(distancias[i][outro_cluster])
                )

        denominador = max(a, b)

        if denominador > 0:
            soma += (b - a) / denominador

    return soma / n


def calcular_silhueta_por_k(
    X,
    k_min=2,
    k_max=10,
    n_init=10
):
    """
    Executa o K-Means para cada k na faixa [k_min, k_max]
    e retorna a silhueta média correspondente.
    Utilizado para visualizar o k ideal pelo pico na curva.
    """

    # Matriz de distâncias calculada uma única vez
    # e reaproveitada para todos os valores de k
    distancias = np.linalg.norm(
        X[:, np.newaxis, :] - X[np.newaxis, :, :],
        axis=2
    )

    ks = list(range(k_min, k_max + 1))
    scores = []

    for k in ks:

        _, _, labels = k_means(
            X,
            k,
            n_init=n_init
        )

        scores.append(
            _calcular_silhueta_com_distancias(
                distancias,
                labels
            )
        )

    return ks, scores


def k_means(
    X,
    k,
    n_init=10
):

    melhor_inercia = np.inf

    melhor_centroides = None
    melhor_clusters = None
    melhor_labels = None

    # Executa o K-Means várias vezes.
    #
    # Cada execução possui uma nova inicialização
    # através do K-Means++.
    for _ in range(n_init):

        centroides, clusters, labels = (
            _k_means_once(X, k)
        )

        inercia = calcular_inercia(
            centroides,
            clusters
        )

        # Guarda somente a melhor solução
        if inercia < melhor_inercia:

            melhor_inercia = inercia

            melhor_centroides = centroides
            melhor_clusters = clusters
            melhor_labels = labels

    return (
        melhor_centroides,
        melhor_clusters,
        melhor_labels
    )