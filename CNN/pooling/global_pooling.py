from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
import numpy as np
feature_map = np.array([
    [1, 3, 2, 9],
    [5, 6, 1, 7],
    [4, 2, 8, 6],
    [3, 5, 7, 2]
],dtype=np.float32).reshape(1,4,4,1)

gm_pool = GlobalMaxPooling2D()
gm_output = gm_pool(feature_map)

ga_pool = GlobalAveragePooling2D()
ga_output = ga_pool(feature_map)

print("Global Max Pooling Output:",gm_output.numpy())
print("Global Average Pooling Output:",ga_output.numpy())
