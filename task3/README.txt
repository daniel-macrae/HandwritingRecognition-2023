# Instructions to test task 3 recognizer

1. If you haven't already created the conda environemnt in tasks 1&2 (the environments for both tasks are identical), create a new conda environment, and install the dependencies:
   
   ```conda env create -f Gr9_HWR_env.yml ```

2. Activate the conda environment (if you haven't already done so)

   ```conda activate Gr9_HWR_env```

3. Run the testing script

    ```python testing_end2end_recognizer.py -i path/to/test/images/```

    It requires an argument for the path of the folder containing the images to be processed ("Test_imgs/" is the placeholder for this directory). Results are written to `/results` folder in the same directory the script is located at. Note that this pipeline will take longer to run than the one we submitted for tasks 1 & 2.



If you don't have yet Anaconda installed in your system you can do so by following these steps:
1. Download Anaconda installer

  ```wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh```

2. Install Anaconda

  ```bash Anaconda3-2023.03-1-Linux-x86_64.sh```

3. Accept license terms
4. Select installation directory
5. Set path variables and restart shell
