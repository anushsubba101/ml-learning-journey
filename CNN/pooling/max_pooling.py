from tensorflow.keras.layers import MaxPooling2D # type: ignore
import numpy as np

feature_map = np.array([
    [1,3,2,9],
    [5,6,1,7],
    [4,2,8,6],
    [3,5,7,2]
]).reshape(1,4,4,1)

max_pool = MaxPooling2D(pool_size=(2,2),strides=2)
output = max_pool(feature_map)
print(output.numpy().reshape(2,2))

