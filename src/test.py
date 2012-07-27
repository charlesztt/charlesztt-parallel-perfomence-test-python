import data_io_test as di
import kmeans as km
import numpy as np
import time

data_x=di.read_test_image('../data/test_image.dat')
label_o=di.read_test_label('../data/test_label.dat')
labels=np.zeros(label_o.shape);

c1=time.time();
km.kmeans(data_x, 10, labels)
print time.time()-c1;
