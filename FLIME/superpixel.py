from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
from functools import partial

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm


from lime import lime_base
from lime.wrappers.scikit_image import SegmentationAlgorithm
# args
args = {"image": r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\img3.png'}

# load the image and apply SLIC and extract (approximately)
# the supplied number of segments
image = cv2.imread(args["image"])
#image = cv2.resize(image,(299,299))
segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=1000, ratio=0.2,
                                                    random_seed=1000)
segments = segmentation_fn(image)
#segments = slic(img_as_float(image), n_segments=50, sigma=5)
print(segments.shape)
# show the output of SLIC
#print(image.shape)
fig = plt.figure('Superpixels')
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()
print("segments:\n", segments)
print("np.unique(segments):", np.unique(segments))
print(segments.shape)
# loop over the unique segment values
count = 0
for (i, segVal) in enumerate(np.unique(segments)):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    #mask[:] = 255;
    mask[segments == segVal] = 255
    #mask[segments == segVal + 4] = 255
    # mask[segments == segVal + 2] = 255
    # mask[segments == segVal + 3] = 255
    # mask[segments == segVal + 4] = 255
    cv2.imshow("Mask", mask)
    #cv2.waitKey(0)
    #cv2.imshow("Applied", np.multiply(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0))
    #print(mask.shape)
    cv2.imwrite('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/superpixel-result/' + str(count) + '.jpg', np.multiply(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0))
    count = count + 1
    #cv2.waitKey(0)
# for (i, segVal) in enumerate(np.unique(segments)):
#     # construct a mask for the segment
#     print("[x] inspecting segment {}, for {}".format(i, segVal))
#     mask = np.zeros(image.shape[:2], dtype="uint8")
#     #mask[(segments != segVal) or (segments != segVal + 1)] = 255
#     #mask[segments != (segVal  (segVal + 1))] = 255
#     mask[:] = 255;
#     mask[segments == segVal] = 0
#     #mask[segments == segVal + 1] = 0
#     # show the masked region
#     cv2.imshow("Mask", mask)
#     cv2.imshow("Applied", np.multiply(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0))
#     cv2.waitKey(0)