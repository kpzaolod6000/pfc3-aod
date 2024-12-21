import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_uniform_depth(rows, cols, rho):
    """
    Genera un mapa de profundidad uniforme para toda la imagen.
    """
    depth_map = np.full((rows, cols), rho)  # Mapa uniforme
    return depth_map

def add_haze_advanced(image, beta, rho, A=1.0):
    """
    Aplica neblina a toda la imagen utilizando un modelo ASM avanzado.

    Args:
        image (numpy array): Imagen de entrada (en formato RGB).
        beta (float): Coeficiente de dispersión atmosférica.
        rho (float): Simulación de la distancia (profundidad).
        A (float): Luz atmosférica global.

    Returns:
        numpy array: Imagen con neblina añadida.
    """
    # Normalizar la imagen
    image = image / 255.0

    # Dimensiones de la imagen
    rows, cols, _ = image.shape

    # Generar mapa de profundidad uniforme
    depth_map = compute_uniform_depth(rows, cols, rho)

    # Calcular la transmisión t(x)
    t_matrix = np.exp(-beta * depth_map)

    # Aplicar el modelo ASM
    A_scaled = np.ones_like(image) * A
    hazy_image = image * t_matrix[:, :, np.newaxis] + A_scaled * (1 - t_matrix[:, :, np.newaxis])

    # Escalar de vuelta al rango [0, 255]
    hazy_image = np.clip(hazy_image, 0, 1)
    return (hazy_image * 255).astype(np.uint8)

# Cargar la imagen
image_path = "paisaje2.jpg"  # Cambia a la ruta de tu imagen
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Parámetros de neblina global
rho = 1.0  # Profundidad uniforme
beta_light = 0.8
beta_medium = 1.0
beta_heavy = 1.4

# Aplicar neblina a toda la imagen
light_haze = add_haze_advanced(image, beta=beta_light, rho=rho, A=1.0)
medium_haze = add_haze_advanced(image, beta=beta_medium, rho=rho, A=1.0)
heavy_haze = add_haze_advanced(image, beta=beta_heavy, rho=rho, A=1.0)

# Visualizar los resultados
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Imagen Original")
plt.imshow(image)
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Light Haze (ASM)")
plt.imshow(light_haze)
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Medium Haze (ASM)")
plt.imshow(medium_haze)
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Heavy Haze (ASM)")
plt.imshow(heavy_haze)
plt.axis("off")

plt.tight_layout()
plt.show()
