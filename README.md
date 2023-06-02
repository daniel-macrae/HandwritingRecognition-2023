# HandwritingRecognition-2023

Made with a virtual environment, with the libraries in requirements.txt

The 'monkbrill', 'monkbrill2', 'IAM-data' and 'image-data' folders you'll have to download yourself.
 



Basic outline of tasks 1 & 2 in the "dssRecognition.py" file

1. Load an image, crop the empty space away
2. Find the optimal rotation (using horizontal peaks)
3. Segment all of the letters, obtaining the boundingboxes for the whole image
4. Find the optimal number of clusters, and use K-means to group the bounding boxes into lines (and sort the lines from left-to-right)
5. Apply the classifier to each letter
6. Write the results into a word file



Notes-to-self:
- add args parser to pick which image or folder to run on (and output folder)
- add debugging option so that it saves the intermediate images somewhere
- also save the results as a txt file because word documents are annoying to find and open