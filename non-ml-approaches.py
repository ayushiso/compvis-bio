import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage.segmentation import chan_vese
from skimage.segmentation import slic
from skimage import color
from skimage.future import graph
from skimage.filters import sobel
from skimage import morphology

from keras import backend as K

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)

  iou = K.eval(iou)
  return iou

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2 * intersection + smooth)/(union + smooth), axis=0)

  dice = K.eval(dice)
  return dice

with open("x_train.pickle", "rb") as input_file:
    x_train = pickle.load(input_file)
    
with open("y_train.pickle", "rb") as input_file:
    y_train = pickle.load(input_file)

with open("x_test.pickle", "rb") as input_file:
    x_test = pickle.load(input_file)
    
with open("y_test.pickle", "rb") as input_file:
    y_test = pickle.load(input_file)

inputs = []

def GMM(img):
    hist, bin_edges = np.histogram(img, bins=60)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    classif = GaussianMixture(n_components=2)
    classif.fit(img.reshape((img.size, 1)))

    threshold = np.mean(classif.means_)
    binary_img = img > threshold

    # check for inverted colors
    if np.sum(binary_img) > 0.60*binary_img.size:
        binary_img = np.logical_not(binary_img)
    
    return binary_img

def chan_vese_calc(img):
    cv = chan_vese(img, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=500,
        dt=0.5, init_level_set="checkerboard", extended_output=True)
    result = cv[1]

    # check for inverted colors
    if np.sum(result) > 0:
        result = 1-result
    return result

def normalized_cut(img):
    labels1 = slic(img, compactness=50, n_segments=1000)
    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, img, kind='avg')
    # print(np.mean(out2))
    # check for inverted colors
    if np.mean(out2) > 100:
        out2 = 200 -out2
    return out2

def region_based(img):
    elevation_map = sobel(img)
    markers = np.zeros_like(img)
    markers[img < 20] = 1
    markers[img > 60] = 2

    # check for inverted colors
    if np.mean(markers) > 1.80:
        markers = np.zeros_like(img)
        markers[img < 80] = 2
        markers[img > 150] = 1

    segmentation = morphology.watershed(elevation_map, markers)
    segmentation -= 1

    return segmentation

# save images 
def save_images(inpt, truth):
    for i in range(10, 60, 10):
        print("img=", i)

        img = inpt.squeeze(-1)[i]
        iname = "input_" + str(i) +".png"
        plt.imsave(iname, img, cmap=plt.cm.gray)

        # gt = 1-truth[i].argmax(axis=-1)
        # fname = "gt_" + str(i) +".png"
        # plt.imsave(fname, gt, cmap=plt.cm.gray)                                                   

        # Gaussian mixture model results
        # binary_img = GMM(img)
        # gname = "gmm_" + str(i) + ".png"
        # plt.imsave(gname, binary_img, cmap=plt.cm.gray)

        # cv = chan_vese_calc(img)
        # cname = "cv_" + str(i) + ".png"
        # plt.imsave(cname, cv, cmap=plt.cm.gray)

        # rb = region_based(img)
        # rname = "rb_" + str(i) + ".png"
        # plt.imsave(rname, rb, cmap=plt.cm.gray)

        nc = normalized_cut(img)
        nname = "ncut_" + str(i) + ".png"
        plt.imsave(nname, nc, cmap=plt.cm.gray)

def pix_acc(y_true, y_pred):
    acc = 0.0
    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)
    for i in range(y_pred.shape[0]):
        acc += np.count_nonzero(y_pred[i] == y_true[i])/(y_pred[i].size)
    return acc/y_pred.shape[0]

def one_hot(a):
    print(a.shape)
    n_values = np.max(a) + 1
    b = np.eye(n_values)[a]
    print(b.shape)
    return b

gmm = []
nc = []
cv = []
rb = []

# save_images(x_test, y_test)

for i in range(100):
    print("item", i)
    img = x_test.squeeze(-1)[i]
    # gmm.append(GMM(img))
    nc_img = normalized_cut(img)
    max_nc = np.max(nc_img)
    nc_img = (nc_img > 0.5*max_nc).astype(int)
    nc.append(nc_img)
    # cv.append(chan_vese(img))
    # rb.append(region_based(img))

# # ab = np.random.randint(2, size=(2,128,128))
# # ab = one_hot(ab)
# gmm = one_hot(np.array(gmm).astype(int))

nc = one_hot(np.array(nc))
# cv = one_hot(np.array(cv).astype(int))
# rb = one_hot(np.array(rb))

# print("GMM:\tPixel acc:{}\tIOU:{}\tDice:{}".format(pix_acc(1-y_test, gmm), iou_coef(1-y_test, gmm), dice_coef(1-y_test, gmm)))
# print("CV:\tPixel acc:{}\tIOU:{}\tDice:{}".format(pix_acc(1-y_test, cv), iou_coef(1-y_test, cv), dice_coef(1-y_test, cv)))
#print("RB:\tPixel acc:{}\tIOU:{}\tDice:{}".format(pix_acc(1-y_test, rb), iou_coef(1-y_test, rb), dice_coef(1-y_test, rb)))
print("NC:\tPixel acc:{}\tIOU:{}\tDice:{}".format(pix_acc(1-y_test, nc), iou_coef(1-y_test, nc), dice_coef(1-y_test, nc)))