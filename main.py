import cv2
import numpy as np
import matplotlib.pyplot as plt

# Завантаження моделі та ваг
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Завантаження назв класів з файлу
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Генерація кольорів для кожного класу
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Завантаження зображення
img = cv2.imread('image.jpg')
height, width, _ = img.shape

# Підготовка вхідних даних для моделі
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

# Визначення знайдених об'єктів
boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

# Використання NMS для усунення накладань
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Показати результати
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i], 2))
    color = [int(c) for c in COLORS[class_ids[i]]]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, f'{label} {confidence}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
