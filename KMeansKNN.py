# Professor
## Mohammed K Jummah
## Professor404@gmail.com
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

####### Importing the image #######
img = cv2.imread('1.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im = img.reshape((-1, 3))
im = np.float32(im)

n_clusers = 7
sub = n_clusers // 4  # for arranging the clusters in the figure 

#### KMeans ####
_, labels, center = cv2.kmeans(im, n_clusers, None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1.0), attempts=10, flags=cv2.KMEANS_PP_CENTERS)

#### KNN ####
(trainX, testX, trainY, testY) = train_test_split(im, labels, test_size=0.75)

knn = cv2.ml.KNearest_create()
knn.train(trainX, cv2.ml.ROW_SAMPLE, np.ravel(trainY,order='C'))
ret, results,neighbours,dist = knn.findNearest(im, k=n_clusers)

score = 100.00 * accuracy_score(labels, results)
print("Accuracy: {:.2f}%".format(score))

### PCA ####
pca = PCA(3)
im = pca.fit_transform(im)

im = im.reshape(img.shape) # reshape the image to the original shape

plt.subplot(sub+1,4,1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')
for i in range(n_clusers):
    mask = np.zeros(im.shape)
    for j in range(im.shape[0]):
        for k in range(im.shape[1]):
            if results[j * im.shape[1] + k] == i:
                mask[j, k] = im[j, k]
    plt.subplot(sub+1, 4, i + 2)
    plt.imshow(mask)
    plt.title('Cluster ' + str(i+1))
    plt.axis('off')
plt.show()
