# Instructions to test task 1 & 2 character recognition pipeline

1. If you haven't already created the conda environemnt in tasks 3 (the environments for both tasks are identical), create a conda environment, and install the dependencies with:
   
   ```conda env create -f Gr9_HWR_env.yml ```

2. Activate the conda environment (if you haven't already done so)

   ```conda activate Gr9_HWR_env```

3. Run the testing script

    ```python dssRecognition.py ./test_images```

    It requires an argument for the path of the folder containing the images to be processed ("test_images/" is the placeholder for this directory). Results are written to a `/results` folder in the same directory the script is located at. 
    
    Additionally, for additional viewing of our code's proceses, a '/debug' folder offers visualisatons of the results of the various steps of the segmentation and classification process, which can be enabled by passing another argument to the bash command (e.g. ```python dssRecognition.py ./test_images True```).



If you don't have yet Anaconda installed in your system you can do so by following these steps:
1. Download Anaconda installer

  ```wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh```

2. Install Anaconda

  ```bash Anaconda3-2023.03-1-Linux-x86_64.sh```

3. Accept license terms
4. Select installation directory
5. Set path variables and restart shell
