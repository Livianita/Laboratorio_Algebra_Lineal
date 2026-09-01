import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataset import build_matrix, load_images


def main():
    # Carga las imagenes y las organiza en una matriz para aplicar PCA.
    _, images = load_images()
    X = build_matrix(images)

    # Se ajusta PCA sin limite de componentes para conocer la varianza
    # retenida al acumular cada posible numero de componentes principales.
    pca_full = PCA()
    pca_full.fit(X)

    varianza_acumulada = pca_full.explained_variance_ratio_.cumsum()
    valores_k = range(1, len(varianza_acumulada) + 1)

    # Valores de varianza explicada solicitados en la actividad.
    niveles_varianza = [0.99, 0.75, 0.3]

    # Para cada nivel de varianza, se identifica y muestra el numero minimo
    # de componentes principales k que se deben conservar.
    print("Verificacion del numero de componentes principales retenidos:")
    for nivel in niveles_varianza:
        pca_nivel = PCA(n_components=nivel)
        pca_nivel.fit(X)
        k = pca_nivel.n_components_
        print(f"Para explained_variance = {nivel}, se retienen k = {k} componentes")

    # Grafica la varianza retenida acumulada en funcion de k.
    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, varianza_acumulada, marker="o", markersize=3)
    plt.xlabel("Numero de componentes principales retenidos k")
    plt.ylabel("Varianza retenida acumulada")
    plt.title("Varianza retenida en funcion del numero de componentes principales")
    plt.grid(True)

    # Marca en la grafica el valor de k correspondiente a cada nivel solicitado.
    for nivel in niveles_varianza:
        pca_nivel = PCA(n_components=nivel)
        pca_nivel.fit(X)
        k = pca_nivel.n_components_

        plt.axhline(y=nivel, color="gray", linestyle="--", linewidth=1)
        plt.axvline(x=k, color="gray", linestyle="--", linewidth=1)
        plt.scatter(k, varianza_acumulada[k - 1], color="red")
        plt.text(k, varianza_acumulada[k - 1], f"  k={k}, var={nivel}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
