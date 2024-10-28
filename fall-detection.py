import cv2 as cv
import numpy as np
from keras.models import load_model

# Desabilitar notação científica
np.set_printoptions(suppress=True)

# Carregar modelo - Teachable Machine
model = load_model("./data/model-classifier/fall_detection_model.h5", compile=False)

# Carregar as legendas
class_names = open("./data/model-classifier/labels.txt", "r").readlines()

# Carregar os arquivos do modelo SSD
ssd_net = cv.dnn.readNetFromCaffe(
    "./data/ssd/MobileNetSSD_deploy.prototxt",
    "./data/ssd/MobileNetSSD_deploy.caffemodel"
)

# Classe de interesse para detecção do modelo SSD - person
SSD_CLASSES = { 15: "person" } 

# Câmera/Webcam
camera = cv.VideoCapture(0)

while True:
    # Imagem da câmera
    ret, image = camera.read()
    if not ret:
        break

    # Preparação da imagem para a entrada (input) do SSD
    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # Blob como entrada (input) no SSD
    ssd_net.setInput(blob)
    detections = ssd_net.forward()

    # Loop para encontrar pessoas
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > 0.5 and class_id in SSD_CLASSES:
            # Extrai as coordenadas da pessoa detectada
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extrai o ROI (region of interest) para detectar pessoa
            person_roi = image[startY:endY, startX:endX]

            # Pré-processamento para o modelo de classificação
            resized = cv.resize(person_roi, (224, 224), interpolation=cv.INTER_AREA)
            resized = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
            resized = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
            resized = (resized / 127.5) - 1

            # Predição usando o modelo de classificação
            prediction = model.predict(resized)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Desenha um retângulo e a legenda em torno do ROI (region of interest)
            cv.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv.putText(image, f"{class_name[2:]}: {confidence_score:.2f}", 
                       (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostrar a imagem em uma janela
    cv.imshow("Webcam Image", image)
    
    if cv.waitKey(33) == ord("q"):
        break

camera.release()
cv.destroyAllWindows()
