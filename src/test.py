import data_io_test as di
import kmeans as km
import numpy as np
import time

data_x=di.read_test_image('../data/test_image.dat')
label_o=di.read_test_label('../data/test_label.dat')
labels=np.zeros(label_o.shape);

data_size=data_x.shape;

centroids=np.zeros((data_size[0],10));

for n in range(0,10):
    centroids[:,n]=data_x[:,np.where(label_o==n)[1]].mean(axis=1)

c1=time.time();
km.kmeans(data_x, 10, labels, centroids)
print time.time()-c1;
