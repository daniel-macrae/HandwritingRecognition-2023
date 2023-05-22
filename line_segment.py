import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the binarized image
image = cv2.imread('D:\home\lucypher\Desktop\HandRec\Project\image-data\image-data\P632-Fg002-R-C01-R01-binarized.jpg', cv2.IMREAD_GRAYSCALE)

# Reduce the image vertically to create a vertical vector
vertical_vector = np.sum(image, axis=1)

# Plot the vertical vector as a histogram
plt.plot(vertical_vector)
plt.title('Vertical Vector')
plt.xlabel('Vertical Axis')
plt.ylabel('Sum')
plt.show()

# Find the peaks in the vertical vector
peaks, _ = find_peaks(vertical_vector, distance=50, prominence=50)

# Extract lines based on peak indices
for i in range(len(peaks) - 1):
    line = image[peaks[i]:peaks[i+1], :]
    line_sum = np.sum(line)
    
    # Set a minimum threshold to filter out insignificant peaks
    threshold = 850000
    
    if line_sum>threshold:
        # Process the extracted line (e.g., save, display, or perform further operations)
        cv2.imshow(f'Line {i+1}', line)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
