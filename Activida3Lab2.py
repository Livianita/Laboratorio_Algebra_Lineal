# PCA aplicado al dataset Optical Recognition of Handwritten Digits de UCI (id=80).
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo


def main():
    # 1. Cargar las 64 caracteristicas de pixeles y la etiqueta de cada digito.
    digitos = fetch_ucirepo(id=80)
    X = digitos.data.features.to_numpy()
    clase_digito = digitos.data.targets["class"].to_numpy()

    # Todos los pixeles usan la misma escala de 0 a 16, por lo que no se
    # requiere estandarizarlos antes de aplicar PCA.
    pca_completo = PCA()
    pca_completo.fit(X)

    # 2. Calcular la varianza acumulada y los componentes para 90% y 95%.
    varianza_acumulada = np.cumsum(pca_completo.explained_variance_ratio_)
    componentes = np.arange(1, len(varianza_acumulada) + 1)
    k_90 = np.argmax(varianza_acumulada >= 0.90) + 1
    k_95 = np.argmax(varianza_acumulada >= 0.95) + 1

    print("Numero original de variables:", X.shape[1])
    print(f"Componentes para retener 90% de varianza: {k_90}")
    print(f"Componentes para retener 95% de varianza: {k_95}")
    print(f"Varianza retenida con 2 componentes: {varianza_acumulada[1] * 100:.2f}%")

    # 3. Graficar la varianza retenida segun el numero de componentes.
    plt.figure(figsize=(8, 5))
    plt.plot(componentes, varianza_acumulada, marker="o", markersize=3)
    plt.axhline(0.90, color="orange", linestyle="--", label="90% de varianza")
    plt.axhline(0.95, color="red", linestyle="--", label="95% de varianza")
    plt.axvline(k_90, color="orange", linestyle="--")
    plt.axvline(k_95, color="red", linestyle="--")
    plt.xlabel("Numero de componentes principales")
    plt.ylabel("Varianza explicada acumulada")
    plt.title("Utilidad de PCA en digitos manuscritos")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Proyectar los digitos en las dos primeras componentes principales.
    pca_2d = PCA(n_components=2)
    datos_2d = pca_2d.fit_transform(X)

    plt.figure(figsize=(9, 7))
    colores = plt.cm.tab10
    for digito in np.unique(clase_digito):
        mascara = clase_digito == digito
        plt.scatter(
            datos_2d[mascara, 0],
            datos_2d[mascara, 1],
            label=str(digito),
            color=colores(digito),
            alpha=0.6,
            s=12,
        )

    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title("Digitos manuscritos en las dos primeras componentes")
    plt.legend(title="Digito", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
