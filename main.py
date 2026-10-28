import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from data_loader import carregar_dataset
from kmeans import k_means, calcular_elbow, calcular_silhueta_por_k


# Ativa ou desativa os métodos de escolha do k ideal
MOSTRAR_ELBOW = True
MOSTRAR_SILHUETA = True

# Faixa de k testada no Elbow Method
K_ELBOW_MIN = 2
K_ELBOW_MAX = 10

# Faixa de k testada no Coeficiente de Silhueta
K_SILHUETA_MIN = 2
K_SILHUETA_MAX = 10

N_INIT = 10


X, features = carregar_dataset(
    "datasets/05_cascas_esfericas_3d_concentricas.csv"
)

centroides, clusters, labels = k_means(
    X,
    k=2,
    n_init=N_INIT
)


# ==========================================
# VERIFICAÇÃO DA QUANTIDADE DE DIMENSÕES
# ==========================================

numero_features = X.shape[1]


if numero_features == 2:

    # ======================================
    # GRÁFICO 2D
    # ======================================

    plt.figure()

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

    plt.xlabel(features[0] if features else "F1")
    plt.ylabel(features[1] if features else "F2")

    plt.grid(True)
    plt.legend()


elif numero_features == 3:

    # ======================================
    # GRÁFICO 3D
    # ======================================

    fig = plt.figure()

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    for i in range(len(X)):

        ax.scatter(
            X[i][0],
            X[i][1],
            X[i][2],
            c=f"C{labels[i]}",
            s=100
        )

    ax.scatter(
        centroides[:, 0],
        centroides[:, 1],
        centroides[:, 2],
        c="black",
        marker="X",
        s=300,
        label="Centroides"
    )

    ax.set_title("K-Means")

    ax.set_xlabel(features[0] if features else "F1")
    ax.set_ylabel(features[1] if features else "F2")
    ax.set_zlabel(features[2] if features else "F3")

    ax.legend()


else:

    # ======================================
    # MAIS DE 3 DIMENSÕES
    # ======================================

    print(
        f"O dataset possui {numero_features} dimensões. "
        "Gerando visualizações de alta dimensionalidade..."
    )

    # --------------------------------------
    # 1. GRÁFICO DE COORDENADAS PARALELAS
    # --------------------------------------
    plt.figure(figsize=(10, 5))
    
    # Plota uma linha para cada ponto do dataset
    for i in range(len(X)):
        plt.plot(
            range(numero_features), 
            X[i], 
            color=f"C{labels[i]}", 
            alpha=0.6 # Deixa a linha um pouco transparente
        )

    # Plota os centroides em destaque
    for i, centroide in enumerate(centroides):
        plt.plot(
            range(numero_features), 
            centroide, 
            color="black", 
            linewidth=3, 
            marker="X", 
            markersize=10
        )

    plt.xticks(range(numero_features), features if features else [f"F{i+1}" for i in range(numero_features)])
    plt.title("Coordenadas Paralelas - K-Means (Centroides em Preto)")
    plt.ylabel("Valores das Features")
    plt.grid(True, linestyle="--", alpha=0.5)


    # --------------------------------------
    # 2. MATRIZ DE DISPERSÃO (PAIRPLOT)
    # --------------------------------------
    fig, axes = plt.subplots(
        numero_features, 
        numero_features, 
        figsize=(12, 12)
    )
    
    fig.suptitle("Matriz de Dispersão (K-Means)", fontsize=16)

    for i in range(numero_features):
        for j in range(numero_features):
            ax = axes[i, j]
            
            if i == j:
                # Na diagonal (feature contra ela mesma), exibe o nome da feature
                ax.text(
                    0.5, 0.5, 
                    features[i] if features else f"F{i+1}", 
                    ha="center", 
                    va="center", 
                    fontsize=12, 
                    weight="bold"
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Plota a dispersão 2D cruzando a feature J (eixo X) com a feature I (eixo Y)
                for idx in range(len(X)):
                    ax.scatter(
                        X[idx, j], 
                        X[idx, i], 
                        c=f"C{labels[idx]}", 
                        s=15, 
                        alpha=0.7
                    )
                
                # Plota os centroides no gráfico 2D
                ax.scatter(
                    centroides[:, j], 
                    centroides[:, i], 
                    c="black", 
                    marker="X", 
                    s=60
                )

            # Limpa os números dos eixos internos para o gráfico não ficar poluído
            if i < numero_features - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)


# ==========================================
# ELBOW METHOD
# ==========================================

if MOSTRAR_ELBOW:

    ks_elbow, inercias_elbow = calcular_elbow(
        X,
        k_min=K_ELBOW_MIN,
        k_max=K_ELBOW_MAX,
        n_init=N_INIT
    )

    plt.figure()

    plt.plot(
        ks_elbow,
        inercias_elbow,
        marker="o",
        linestyle="-",
        color="tab:blue"
    )

    plt.xlabel("k (número de clusters)")
    plt.ylabel("Inércia")
    plt.title("Elbow Method - Escolha do k Ideal")

    plt.grid(True)


# ==========================================
# COEFICIENTE DE SILHUETA
# ==========================================

if MOSTRAR_SILHUETA:

    ks_silhueta, scores_silhueta = calcular_silhueta_por_k(
        X,
        k_min=K_SILHUETA_MIN,
        k_max=K_SILHUETA_MAX,
        n_init=N_INIT
    )

    plt.figure()

    plt.plot(
        ks_silhueta,
        scores_silhueta,
        marker="o",
        linestyle="-",
        color="tab:green"
    )

    plt.xlabel("k (número de clusters)")
    plt.ylabel("Coeficiente de Silhueta médio")
    plt.title("Coeficiente de Silhueta - Escolha do k Ideal")

    plt.grid(True)

plt.show()