import cv2
import numpy as np
import os

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

# Configuración de carpetas
input_folder = "/home/pytorch/data/images_gun/train2024"
output_folder = "results_gun/train2024"
os.makedirs(output_folder, exist_ok=True)

# Parámetros de neblina global
rho = 1.0  # Profundidad uniforme
beta_values = {
    "light_haze": 0.4,
    "medium_haze": 0.6,
    "heavy_haze": 1.0
}

# Procesar todas las imágenes de la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        image_path = os.path.join(input_folder, filename)
        
        # Cargar y convertir la imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Aplicar neblina y guardar las imágenes
        for haze_level, beta in beta_values.items():
            hazy_image = add_haze_advanced(image, beta=beta, rho=rho, A=1.0)
            
            # Crear subcarpeta según el tipo de niebla
            haze_folder = os.path.join(output_folder, haze_level)
            os.makedirs(haze_folder, exist_ok=True)
            
            # Guardar la imagen con el nombre original (sin modificar el nombre del archivo)
            output_path = os.path.join(haze_folder, filename)
            cv2.imwrite(output_path, cv2.cvtColor(hazy_image, cv2.COLOR_RGB2BGR))
            print(f"Imagen guardada: {output_path}")

print(f"Proceso completado. Imágenes con neblina guardadas en subcarpetas dentro de '{output_folder}'")
