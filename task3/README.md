# Instructions to test task 3 recognizer

1. Install environment requirements
   
   ```pip install -r requirements.txt```

2. Run the testing script

    ```python testing_end2end_recognizer.py test_imgs/```

    It requires an argument for the path of the folder containing the images to be processed. Results are written to `/results` folder in the same directory the script is located at. 

    The script will download the model weights (~ 4GB) as a zip file and unpack them, this might take a few minutes. Make sure you have around 8GB of free storage.
