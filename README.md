# Instructions to test task 3 recognizer

1. Create a conda environment, and install the dependencies:
   
   ```conda env create -f Gr9_HWR1_2env.yml ```

2. Activate the conda environment

   ```conda activate Gr9_HWR1_2env```

2. Run the testing script

    ```python dssRecognition.py ./test_images```

    It requires an argument for the path of the folder containing the images to be processed ("test_images/" is the placeholder for this directory). Results are written to a `/results` folder in the same directory the script is located at. Additioanlly, a '/debug' folder offers visualisatons of the results of the various steps of the segmentation and classification process.
    


Basic outline of tasks 1 & 2 in the "dssRecognition.py" file

1. Load an image, crop the empty space away
2. Find the optimal rotation (using horizontal peaks)
3. Segment all of the letters, obtaining the boundingboxes for the whole image
4. Find the optimal number of clusters, and use K-means to group the bounding boxes into lines (and sort the lines from left-to-right)
5. Apply the classifier to each letter
6. Write the results into a word file

There's an option to enable/disable saving debugging outputs to the 'debug' folder at the very end, in the "if __name__ == "__main__":" thing

