# Librerias
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# 1. Cargar imagen
img = Image.open("mi_foto.jpeg")  # Formatos aceptados: jpeg, png o jpg.
img_gray = img.convert("L")  # Convierte la imagen a escala de grises.
img_resized = img_gray.resize((512, 512), Image.LANCZOS)
img_array = np.array(img_resized, dtype=float)


# 2. Aplicar PCA para compresion
# Cada fila es una muestra y cada columna es una caracteristica.

# PCA que conserva al menos el 70% de la varianza.
pca_70 = PCA(n_components=0.70)
imagen_comprimida_70 = pca_70.fit_transform(img_array)

# PCA que conserva al menos el 99% de la varianza.
pca_99 = PCA(n_components=0.99)
imagen_comprimida_99 = pca_99.fit_transform(img_array)

# PCA que conserva al menos el 80% de la varianza. Con 70% en la priemra corrida no se ve bien la imagen
pca_80 = PCA(n_components=0.80)
imagen_comprimida_80 = pca_80.fit_transform(img_array)

# PCA que conserva al menos el 90% de la varianza. agrego 90% aun con 80% no se ve bien la imagen
pca_90 = PCA(n_components=0.90)
imagen_comprimida_90 = pca_90.fit_transform(img_array)

# 3. Reconstruccion
# Recupera una aproximacion de la imagen original desde la version comprimida.
imagen_reconstruida_70 = pca_70.inverse_transform(imagen_comprimida_70)
imagen_reconstruida_99 = pca_99.inverse_transform(imagen_comprimida_99)
imagen_reconstruida_80 = pca_80.inverse_transform(imagen_comprimida_80)
imagen_reconstruida_90 = pca_90.inverse_transform(imagen_comprimida_90)


# 4. Visualizacion
plt.figure(figsize=(15, 5))

plt.subplot(1, 5, 1)
plt.imshow(img_array, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(imagen_reconstruida_70, cmap="gray", vmin=0, vmax=255)
plt.title(f"Reconstruccion 70%\nk = {pca_70.n_components_}")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(imagen_reconstruida_99, cmap="gray", vmin=0, vmax=255)
plt.title(f"Reconstruccion 99%\nk = {pca_99.n_components_}")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(imagen_reconstruida_80, cmap="gray", vmin=0, vmax=255)
plt.title(f"Reconstruccion 80%\nk = {pca_80.n_components_}")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(imagen_reconstruida_90, cmap="gray", vmin=0, vmax=255)
plt.title(f"Reconstruccion 90%\nk = {pca_90.n_components_}")
plt.axis("off")

plt.tight_layout()
plt.show()

# 5. Mostrar informacion
print("Informacion de compresion con PCA")

print("\nCaso: 70% de varianza")
print("Dimension de la imagen original:", img_array.shape)
print("Dimension de la representacion comprimida:", imagen_comprimida_70.shape)
print("Numero de componentes principales k:", pca_70.n_components_)
print(
    "Porcentaje de varianza retenida:",
    pca_70.explained_variance_ratio_.sum() * 100,
    "%",
)

print("\nCaso: 99% de varianza")
print("Dimension de la imagen original:", img_array.shape)
print("Dimension de la representacion comprimida:", imagen_comprimida_99.shape)
print("Numero de componentes principales k:", pca_99.n_components_)
print(
    "Porcentaje de varianza retenida:",
    pca_99.explained_variance_ratio_.sum() * 100,
    "%",
)

print("\nCaso: 80% de varianza")
print("Dimension de la imagen original:", img_array.shape)
print("Dimension de la representacion comprimida:", imagen_comprimida_80.shape)
print("Numero de componentes principales k:", pca_80.n_components_)
print(
    "Porcentaje de varianza retenida:",
    pca_80.explained_variance_ratio_.sum() * 100,
    "%",
)
print("\nCaso: 90% de varianza")
print("Dimension de la imagen original:", img_array.shape)
print("Dimension de la representacion comprimida:", imagen_comprimida_90.shape)
print("Numero de componentes principales k:", pca_90.n_components_)
print(
    "Porcentaje de varianza retenida:",
    pca_90.explained_variance_ratio_.sum() * 100,
    "%",
)