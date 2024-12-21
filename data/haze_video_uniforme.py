import cv2
import numpy as np
import os

def compute_uniform_depth(rows, cols, rho):
    """
    Genera un mapa de profundidad uniforme para toda la imagen.
    """
    depth_map = np.full((rows, cols), rho)  # Mapa uniforme
    return depth_map

def add_haze_advanced(frame, beta, rho, A=1.0):
    """
    Aplica neblina a un frame utilizando un modelo ASM avanzado.

    Args:
        frame (numpy array): Frame de entrada (en formato RGB).
        beta (float): Coeficiente de dispersión atmosférica.
        rho (float): Simulación de la distancia (profundidad).
        A (float): Luz atmosférica global.

    Returns:
        numpy array: Frame con neblina añadida.
    """
    # Normalizar el frame
    frame = frame / 255.0

    # Dimensiones del frame
    rows, cols, _ = frame.shape

    # Generar mapa de profundidad uniforme
    depth_map = compute_uniform_depth(rows, cols, rho)

    # Calcular la transmisión t(x)
    t_matrix = np.exp(-beta * depth_map)

    # Aplicar el modelo ASM
    A_scaled = np.ones_like(frame) * A
    hazy_frame = frame * t_matrix[:, :, np.newaxis] + A_scaled * (1 - t_matrix[:, :, np.newaxis])

    # Escalar de vuelta al rango [0, 255]
    hazy_frame = np.clip(hazy_frame, 0, 1)
    return (hazy_frame * 255).astype(np.uint8)

# Configuración de entrada y salida
input_video = "video.mp4"  # Archivo de video de entrada
output_folder = "output_videos_uniform"  # Carpeta para guardar videos con neblina
os.makedirs(output_folder, exist_ok=True)

# Parámetros de neblina global
rho = 1.0  # Profundidad uniforme
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

# Aplicar neblina para cada nivel
for haze_level, beta in beta_values.items():
    output_video_path = os.path.join(output_folder, f"output_{haze_level}.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Procesando video con nivel de neblina: {haze_level}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir frame a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Aplicar neblina
        hazy_frame = add_haze_advanced(frame, beta=beta, rho=rho, A=1.0)

        # Convertir de vuelta a BGR para guardar
        hazy_frame = cv2.cvtColor(hazy_frame, cv2.COLOR_RGB2BGR)
        out.write(hazy_frame)

    # Liberar el video writer para este nivel de neblina
    out.release()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar el video al primer frame

cap.release()
print(f"Proceso completado. Videos con neblina guardados en la carpeta '{output_folder}'")
