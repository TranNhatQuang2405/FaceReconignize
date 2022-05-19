from pyexpat import model
from mtcnn.mtcnn import MTCNN
from PIL import Image
from Detection import draw_image_with_boxes
from getModel import Model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

model = Model()

label = {0: "Phuc", 1: "Nguyen", 2: "Nhan", 3: "Phat", 4: "Phuong", 5: "Quang"}

stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
size = 224
batch_size = 64
channels = 3

transformer = T.Compose([
    T.Resize(size),
    T.CenterCrop(size),
    T.ToTensor(),
    T.Normalize(*stats)
])


def denormal(image):
    image = image.numpy().transpose(1, 2, 0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def denormalize(x, mean=stats[0], std=stats[1]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


def Predict(image):
    pixels = np.asarray(image)
    imageResult = pixels.copy()
    detector = MTCNN()
    faces = detector.detect_faces(pixels)
    for result in faces:
        x1, y1, width, height = result['box']
        x2, y2 = x1 + width, y1 + height

        image = Image.fromarray(pixels[y1:y2, x1:x2])
        image.show()

        image = transformer(image).float()
        image = image.unsqueeze_(0)
        outs = model(image)
        index = outs.data.numpy().argmax()
        _, preds = torch.max(outs, dim=1)

        print(label[index])
        imageResult = cv2.rectangle(
            imageResult, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        imageResult = cv2.putText(imageResult, label[index], (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    return Image.fromarray(imageResult)


# filename = './image/test/276157360_1176465043208104_4240594520182539077_n.jpg'

# img = Image.open(filename)
# pixels = np.asarray(img)

# # pixels = plt.imread(filename)
# print("BBBB: ", pixels)

# detector = MTCNN()
# faces = detector.detect_faces(pixels)
# for result in faces:
#     x1, y1, width, height = result['box']
#     x2, y2 = x1 + width, y1 + height
#     image = Image.fromarray(pixels[y1:y2, x1:x2])
#     image.show()

#     image = transformer(image).float()
#     image = image.unsqueeze_(0)
#     outs = model(image)
#     index = outs.data.numpy().argmax()

#     _, preds = torch.max(outs, dim=1)
#     print(label[index])
