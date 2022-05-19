# extract and plot each detected face in a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
import os


def draw_faces(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot each face as a subplot
    for i in range(len(result_list)):
        # get coordinates
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        # define subplot
        pyplot.subplot(1, len(result_list), i+1)
        pyplot.axis('off')
        # plot face
        pyplot.imshow(data[y1:y2, x1:x2])
    # show the plot
    pyplot.show()


def save_face(result_list, data, folder, i):
    x1, y1, width, height = result_list['box']
    x2, y2 = x1 + width, y1 + height
    if not os.path.exists("data/{}".format(folder)):
        os.makedirs("data/{}".format(folder))
    # pyplot.imsave("data/{}/{}_{}.jpg".format(folder,
    #               folder, i), data[y1:y2, x1:x2])

    pyplot.imsave("data/{}/{}_{}.jpg".format(folder,
                  folder, i), cv2.cvtColor(data[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))


def get_face(frame):
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    return faces


def draw_image_with_boxes(filename, result_list):
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
    pyplot.show()
