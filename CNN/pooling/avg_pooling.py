import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D

feature_map = np.array([
    [1,3,2,9],
    [5,6,1,7],
    [4,2,8,6],
    [3,5,7,2]
    ],dtype=np.float32).reshape(1,4,4,1)

avg_pooling = AveragePooling2D(pool_size=(2,2),strides=2)
output = avg_pooling(feature_map)
print(output.numpy().reshape(2,2))