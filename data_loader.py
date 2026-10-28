import numpy as np

def _tem_cabecalho(linha):
    """
    Verifica se a primeira linha do arquivo é um cabeçalho.
    Retorna True se houver alguma string (não numérica),
    ou False se todos os valores forem numéricos.
    """
    tokens = linha.strip().split(",")

    for token in tokens:
        try:
            float(token.strip())
        except ValueError:
            return True

    return False


def carregar_dataset(caminho):

    with open(caminho, "r") as arquivo:
        primeira_linha = arquivo.readline().strip()

    # Se a primeira linha contém strings, é um cabeçalho;
    # caso contrário, é um dado (dataset não rotulado).
    if _tem_cabecalho(primeira_linha):

        features = primeira_linha.split(",")
        skiprows = 1

    else:

        features = []
        skiprows = 0

    # Carrega somente os dados numéricos
    X = np.loadtxt(
        caminho,
        delimiter=",",
        skiprows=skiprows
    )

    return X, features