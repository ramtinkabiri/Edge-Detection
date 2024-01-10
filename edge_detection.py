import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Read the image
I = plt.imread('sample.jpg')
I = np.mean(I, axis=2)

# Define the filters
filters = [
    np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]]),
    np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
    np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).T,
    np.fliplr(np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])),
    np.flipud(np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])),
    np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
    np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
]

# Apply filters
J = np.zeros_like(I)
for filt in filters:
    J = np.maximum(np.abs(convolve(I, filt)), J)

# Thresholding
th = 40
J[J < th] = 0
J[J >= th] = 1

# Display the original image
plt.subplot(1,2,1)
plt.imshow(I, cmap='gray', origin='upper', extent=(0, I.shape[1], I.shape[0], 0))
plt.title('Original Image')
# Display the result
plt.subplot(1,2,2)
plt.imshow(J, cmap='gray', origin='upper', extent=(0, J.shape[1], J.shape[0], 0))
plt.title('Edge Detection Result')
plt.show()


