from sklearn.decomposition import PCA

from dataset import build_matrix, load_images


def main():
    _, images = load_images()
    X = build_matrix(images)
    n = X.shape[1]

    explained_variance = 0.99
    pca = PCA(explained_variance)
    pca.fit(X)

    z = pca.transform(X)
    Vreduce = pca.components_.T
    K = pca.n_components_

    print("Los datos originales tienen dimension", X.shape)
    print("Los datos comprimidos tienen dimension", z.shape)
    print(
        "El numero de componentes principales K es",
        K,
        "que retienen el",
        explained_variance * 100,
        "% de la varianza",
    )
    print("El tam. de Vreduce (matriz de eigenvectors) es", Vreduce.shape)
    print("PCA consigue reducir el tamano en disco al", K / n * 100, "% de su tam. original")


if __name__ == '__main__':
    main()
