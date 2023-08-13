# Handwriting Recognition

This repository contains code for three handwriting recognition tasks:
1. Character segmentation on the Dead Sea Scrolls (DSS), using morphological operations.
2. Character recognition of the segmented DSS characters, using a lightweight LeNet model.
3. Line recognition on a subset of the IAM dataset, using a large end-to-end TrOCR model.

The code for these tasks are grouped into two subfolders, one for tasks 1 and 2, and one for task 3.


![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)


### Instructions


<details>
<summary>Installation</summary>
<br>

<details>
<summary>Installing Anaconda</summary>
<br>

If you don't have yet Anaconda installed in your system you can do so by following these steps:


1. Download Anaconda installer

  ```wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh```

2. Install Anaconda

  ```bash Anaconda3-2023.03-1-Linux-x86_64.sh```

3. Accept license terms
4. Select installation directory
5. Set path variables and restart shell

</details>

1. Create a conda environment and install the necessary dependencies (for all tasks) with:
   
   ```conda env create -f Gr9_HWR_env.yml ```

2. Activating the conda environment

   ```conda activate Gr9_HWR_env```

3. Because the model weights for task 3 are large (~4GB), we opted to submit the code for both tasks via Google Drive. A folder containing a .zip file for each task can be found at:

    https://drive.google.com/drive/folders/1zR3Mf0Bp1QORfLGXWtwzJPdShvawR3Zu?usp=sharing



</details>

<details>
<summary>Datasets</summary>
<br>

For both tasks, the datasets must consist of a folder of images. For task 3, the code is designed to run on a folder of images sourced from the IAM dataset. For tasks 1 and 2, this is the case for binarised images of the Dead Sea Scrolls, which we unfortunetly cannot provide a link to.

</details>

<details>
<summary>Running Tasks 1 & 2</summary>
<br>

1. Activate the conda environment (if you haven't already done so)

   ```conda activate Gr9_HWR_env```

2. Run the testing script

    ```python dssRecognition.py ./test_images```

    It requires an argument for the path of the folder containing the images to be processed ("test_images/" is the placeholder for this directory). 
    Results are written to a `/results` folder in the same directory the script is located at. 
    
    Additionally, for extra viewing of our code's proceses, a '/debug' folder offers visualisatons of the results of the various steps of the segmentation and classification process, which can be enabled by passing another argument to the bash command (e.g. ```python dssRecognition.py ./test_images True```).


</details>


<details>
<summary>Running Task 3</summary>
<br>

1. Activate the conda environment (if you haven't already done so)

   ```conda activate Gr9_HWR_env```


2. Run the testing script (making sure you have downloaded the model weights provided in the installation instructions)

    ```python testing_end2end_recognizer.py -i path/to/test/images/```

    It requires an argument for the path of the folder containing the images to be processed ("Test_imgs/" is the placeholder for this directory). Results are written to `/results` folder in the same directory the script is located at. Note that this pipeline will take noticably longer to run than that of tasks 1 & 2 (a progress bar in the terminal will display how long it is expected to take).


</details>



![green-divider](https://user-images.githubusercontent.com/7065401/52071924-c003ad80-2562-11e9-8297-1c6595f8a7ff.png)



