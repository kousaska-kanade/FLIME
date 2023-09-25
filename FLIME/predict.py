import numpy as np
from PIL import Image

from facenet import Facenet

if __name__ == "__main__":
    model = Facenet()

    image_1 = Image.open(image_1)

    image_2 = Image.open(image_2)

    probability = model.detect_image(image_1, image_2)
    print(probability)
