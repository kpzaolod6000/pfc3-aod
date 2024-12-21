import cv2
import numpy as np
import os
from noise import pnoise2

def generate_perlin_noise(rows, cols, frame_index, scale=100, speed=0.05):
    """
    Genera un mapa de ruido Perlin para simular el movimiento de la niebla.
    Args:
        rows (int): Número de filas.
        cols (int): Número de columnas.
        frame_index (int): Índice del frame actual (para animación).
        scale (float): Escala del ruido Perlin (tamaño de las "partículas").
        speed (float): Velocidad de movimiento de las partículas.

    Returns:
        numpy array: Mapa de ruido Perlin normalizado.
    """
    noise = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            noise[i][j] = pnoise2(i / scale, (j + frame_index * speed) / scale, octaves=3)

    # Normalizar entre [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

def add_haze_with_motion(frame, beta, rho, frame_index, A=1.0):
    """
    Aplica neblina con partículas en movimiento a un frame.
    Args:
        frame (numpy array): Frame de entrada (en formato RGB).
        beta (float): Coeficiente de dispersión atmosférica.
        rho (float): Simulación de la distancia (profundidad).
        frame_index (int): Índice del frame actual.
        A (float): Luz atmosférica global.

    Returns:
        numpy array: Frame con neblina en movimiento añadido.
    """
    frame = frame / 255.0
    rows, cols, _ = frame.shape

    # Generar mapa de profundidad dinámico usando ruido Perlin
    depth_map = generate_perlin_noise(rows, cols, frame_index, scale=200, speed=0.2) * rho

    # Calcular la transmisión t(x)
    t_matrix = np.exp(-beta * depth_map)

    # Aplicar el modelo ASM con partículas en movimiento
    A_scaled = np.ones_like(frame) * A
    hazy_frame = frame * t_matrix[:, :, np.newaxis] + A_scaled * (1 - t_matrix[:, :, np.newaxis])

    # Escalar de vuelta al rango [0, 255]
    hazy_frame = np.clip(hazy_frame, 0, 1)
    return (hazy_frame * 255).astype(np.uint8)

# Configuración de entrada y salida
input_video = "video.mp4"  # Archivo de video de entrada
output_folder = "output_videos"  # Carpeta para guardar videos con neblina
os.makedirs(output_folder, exist_ok=True)

# Parámetros de neblina global
rho = 1.0  # Profundidad máxima
beta_values = {
    "light_haze": 0.4,
    "medium_haze": 0.6,
    "heavy_haze": 1.0
}

# Procesar el video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"No se pudo abrir el video: {input_video}")
    exit()

# Obtener propiedades del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Generar un video para cada nivel de neblina
for haze_level, beta in beta_values.items():
    output_video_path = os.path.join(output_folder, f"output_{haze_level}.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Procesando video con nivel de neblina: {haze_level}")
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir frame a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Aplicar neblina con movimiento
        hazy_frame = add_haze_with_motion(frame, beta=beta, rho=rho, frame_index=frame_index, A=1.0)

        # Convertir de vuelta a BGR para guardar
        hazy_frame = cv2.cvtColor(hazy_frame, cv2.COLOR_RGB2BGR)
        out.write(hazy_frame)

        frame_index += 1

    # Liberar el video writer para este nivel de neblina
    out.release()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar el video al primer frame

cap.release()
print(f"Proceso completado. Videos con neblina guardados en la carpeta '{output_folder}'")
