import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))
	
def relu(x): #ReLU激活函数的实现
    return np.maximum(0,x)
	
def identity_function(x):#输出用恒等函数表示
    return x

def softmax(a):#实现softmax函数，用于分类值得计算，表示概率
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
    
