# Instructions to test task 3 recognizer

1. Create a conda environment, and install the dependencies:
   
   ```conda env create -f Gr9_HWR3env.yml ```

2. Activate the conda environment

   ```conda activate Gr9_HWR3env```

2. Run the testing script

    ```python testing_end2end_recognizer.py test_imgs/```

    It requires an argument for the path of the folder containing the images to be processed ("Test_imgs/" is the placeholder for this directory). Results are written to `/results` folder in the same directory the script is located at. 

    The script will download the model weights (~ 4GB) as a zip file and unpack them, this might take a few minutes. Make sure you have around 8GB of free storage.



If you don't have yet Anaconda installed in your system you can do so by following these steps:
1. Download Anaconda installer

  ```wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh```

2. Install Anaconda

  ```bash Anaconda3-2023.03-1-Linux-x86_64.sh```

3. Accept license terms
4. Select installation directory
5. Set path variables and restart shell
