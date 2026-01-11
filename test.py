import numpy as np

arr = np.arange(10).reshape(2, 5)

# lưu
np.save("arr.npy", arr)

# đọc
arr2 = np.load("arr.npy")
print(arr2.shape, arr2.dtype)
