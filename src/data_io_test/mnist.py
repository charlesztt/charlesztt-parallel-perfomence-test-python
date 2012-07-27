import sys
import numpy as np

def read_train_label(filename):
    f=open(filename,'r');
    f.read(8)
    train_label=np.zeros((1,60000))
    for n in range(0,60000):
        train_label[0,n]=ord(f.read(1))
    f.close()
    return train_label;

def read_train_image(filename):
    f=open(filename,'r');
    f.read(16)
    train_image=np.zeros((784,60000))
    for m in range(0,60000):
        for n in range (0, 784):
            train_image[n,m]=ord(f.read(1))
    f.close()
    return train_image;

def read_test_label(filename):
    f=open(filename,'r')
    f.read(8)
    test_label=np.zeros((1,10000))
    for n in range(0,10000):
        test_label[0,n]=ord(f.read(1))
    f.close()
    return test_label;

def read_test_image(filename):
    f=open(filename,'r')
    f.read(16)
    test_image=np.zeros((784,10000));
    for m in range(0,10000):
        for n in range (0, 784):
            test_image[n,m]=ord(f.read(1))
    f.close();
    return test_image;
