##################################################################################
# --= Pacotes Utilizados =--
# Ambiente de Desenvolvimento
import os

# Matemática
import math

# Opencv
import cv2

# YOLO da ultralytics
from ultralytics import YOLO
##################################################################################

##################################################################################
# --= Aplicação do YOLOv9 na Detecção da Íris
 
# Captura do Vídeo
cap = cv2.VideoCapture("03.videos/video.mp4")

# Instanciando o modelo YOLOv9 (treinado com o conjunto de dados)
modelo = YOLO('04.modelo/best.pt')

def create_detection_dir(base_dir):
    """ Cria e retorna o diretório para salvar a detecção atual """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    max_index = 0
    for subdir in subdirs:
        if subdir.startswith("detection_"):
            index = int(subdir.split("_")[-1])
            if index > max_index:
                max_index = index
    
    new_dir = f"detection_{max_index + 1}"
    full_path = os.path.join(base_dir, new_dir)
    os.makedirs(full_path)
    
    return full_path

def main():
    # Tenta executar o bloco de código principal
    try:
        # Inicializa a captura de vídeo do arquivo especificado
        cap = cv2.VideoCapture("03.videos/video.mp4")

        # Verifica se o vídeo foi aberto corretamente
        if not cap.isOpened():
            raise ValueError("Erro ao abrir o arquivo de vídeo.")
        
        # Carrega o modelo YOLOv9 treinado
        modelo = YOLO('04.modelo/best.pt')
        
        # Cria diretório para a detecção atual
        detection_dir = create_detection_dir("detections")
        video_path = os.path.join(detection_dir, 'detected.mp4')

        # Preparação para gravação do vídeo de recortes
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define o codec
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (224, 224))

        # Loop infinito para processar cada frame do vídeo
        while True:
            # Lê o próximo frame do vídeo
            success, img = cap.read()

            # Se o frame não for lido corretamente, sai do loop
            if not success:
                break

            # Executa a detecção usando o modelo YOLO no frame atual
            detections = modelo(img, stream=True, conf=0.85)

            for r in detections:
                # Acessa as caixas delimitadoras das detecções
                boxes = r.boxes

                for box in boxes:
                    # Extrai e converte as coordenadas da caixa para inteiros
                    x0, y0, x1, y1 = map(int, box.xyxy[0])

                    # Recorta a região da detecção
                    crop_img = img[y0:y1, x0:x1]

                    # Redimensiona o recorte para 224x224 pixels
                    resized_crop = cv2.resize(crop_img, (224, 224))

                    # Grava o recorte no vídeo
                    out.write(resized_crop)

                    # Exibe o recorte redimensionado em tempo real
                    cv2.imshow("ROI Detection", resized_crop)

            # Permite interromper o loop pressionando 'q'
            if cv2.waitKey(1) == ord('q'):
                break

    # Captura exceções específicas com uma mensagem de erro
    except ValueError as e:
        print(e)

    # Garante que os recursos sejam liberados adequadamente
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()