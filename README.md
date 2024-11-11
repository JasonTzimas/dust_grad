# Image Colorization and Morphological filtering based Instance Segmentation

<p align="center">
  <img src="Images/OpenCV_logo_no_text.png" alt="Image description" width="300" height="300">
</p>

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/username/project/actions)

## Description
1. **_PartA_:**
  Educational Code that implements Image colorization from scratch by first cropping 3 images of different channels from a single image, aligns them using Normalized Cross Correlation (NCC) and then combines them in different permutations to give the final RGB Image.

2. **_PartB_:**
  Educational Code that implements Instance Segmentations of multiple well separated objects within an image. The first part of the process is to perform [Otsu's thresholding](https://en.wikipedia.org/wiki/Otsu%27s_method) to get a first segmentation of the objects. Then, morphological filtering is performed to fill any holes/gaps and refine edges. Finally, a Connected-Components Algorithm returns the separated object instance masks.
 


## Table of Contents
- [Part A](#Part_A)
  - [Frame Detection](#frame-detection)
  - [Template Matching and Normalized Cross Correlation (NCC)](#template-matching-and-normalized-cross-correlation-ncc)
  - [Results](#results)
  - [Installation](#installation)
  - [Execution](#execution)
- [Part B](#part-b)
  - [An Input Sample](#an-input-sample)
  - [Otsu's thresholding](#otsus-thresholding)
  - [Morphological Filtering](#morphological-filtering)
  - [Connected Components](#connected-components)
  - [Hue moments, dominant orientations and Centroids](#hue-moments-dominant-orientations-and-centroids)
- [License](#license)

## Part_A

### Frame Detection

The input images are of the following form:

<p align="center">
  <img src="Part_A/InputImages/01112v.jpg" alt="Image 1" width="200" style="border: 2px solid black; margin-right: 10px;">
  <img src="Part_A/InputImages/00125v.jpg" alt="Image 1" width="200" style="border: 2px solid black; margin-right: 10px;">
  <img src="Part_A/InputImages/00149v.jpg" alt="Image 1" width="200" style="border: 2px solid black; margin-right: 10px;">
</p>

The borders coordinates are detected using pixel-value histograms along dimensions x and y. The coordinates are then saved so that we can perform appropriate cropping

### Template Matching and Normalized Cross Correlation (NCC)

After having cropped the images we perform zero-padding and we get the matching coordinates by finding the maximum NCC score while sliding one image on top of the other across x-y axes.

For a given alignment the NCC metric is given by:

<div align="center" style="font-size: 20px;">

$$
NCC = \sum_{i=1}^{W} \sum_{j=1}^{H} \frac{I_1(i, j) - \mu_1}{\sigma_1} * \frac{I_2(i, j) - \mu_2}{\sigma_2}
$$

</div>

, where: 
<div align="center" style="font-size: 12px;">
  
$$
H, W: \text{are the image Height and Width}
$$

</div>

<div align="center" style="font-size: 12px;">
  
$$I_1, I_2: \text{are the two Images}$$

</div>

<div align="center" style="font-size: 12px;">
  
$$ \mu_1, \mu_2, \sigma_1, \sigma_2: \text{are the two Image mean and std values}$$

</div>

### Results

Having found the matching image coordinates we append them together to from a single RGB 3-Channel Image. A sample results is as follows:

<p align="center">
  <img src="Images/rgb55.png" alt="Image description" width="300" height="300">
</p>

### Installation

First create a virual environment by running:

1. Using _Conda_:
  ```bash
  conda create <your_environment_name>
  ```
2. Using _venv_
```bash
python -m venv <your_environment_name>
```

Then, activate your environment:

1. Using _Conda_:
  ```bash
  conda activate <your_environment_name>
  ```

2. Using _venv_:
   ```bash
   cd <your_environment_name>/bin
   source ./activate ## For Linux
   cd ../..
   ```
   or
   ```bash
    cd <your_environment_name>/Scripts
    activate  ## For Windows
    cd ../..
   ```

 Finally, install the requirements.txt file:

 ```bash
 pip install -r requirements.txt
 ```

### Execution

Run:
```bash
python get_rgb.py PartA/InputImages/01112v.jpg <your_output_folder>
```
To reproduce the result above or

```bash
python get_rgb.py path/to/your/image output_folder_name
```
if you want to provide your own image.


## Part B

### An Input sample

<p align="center">
  <img src="Part_B/can.jpg" alt="Image description"  height="300">
</p>

### Otsu's thresholding

The first part of instance segmentation is using Otsu's thresholding method to get a pixel-intensity threshold, fine tuned in a way that leads to maximum cluster separation

The output image after perform Otsu's thresholding is the following:

<p align="center">
  <img src="Images/thresh_otsus.png" alt="Image description"  height="300">
</p>

### Morphological Filtering

We then perform Morphological filtering by means of erosion and dilation to fill in any holes and give the thresholded values a high connectivity. The result of the above filtering process is as follows:

<p align="center">
  <img src="Images/Binary_after.png" alt="Image description"  height="300">
</p>

### Connected Components

Then, a connected-component algorithm using either 4-connectivity or 8-connectivity kernels is implemented to identify the distinct objects present in the image, leading to the following masks:

<p align="center">
  <img src="Images/labeled_image.png" alt="Image description"  height="300">
</p>

<p align="center">
  <img src="Images/labeled_vs_original.png" alt="Image description"  height="300">
</p>

### Hue moments, dominant orientations and Centroids

Hue moments are used to identify the main object axes and centroids as follows:

<p align="center">
  <img src="Images/centroids_rotations.png" alt="Image description"  height="300">
</p>


### Installation

First create a virual environment by running:

1. Using _Conda_:
  ```bash
  conda create <your_environment_name>
  ```
2. Using _venv_
```bash
python -m venv <your_environment_name>
```

Then, activate your environment:

1. Using _Conda_:
  ```bash
  conda activate <your_environment_name>
  ```

2. Using _venv_:
   ```bash
   cd <your_environment_name>/bin
   source ./activate ## For Linux
   ```
   or
   ```bash
    cd <your_environment_name>/Scripts
    ./activate  ## For Windows
   ```

 Finally, install the requirements.txt file:

 ```bash
 pip install -r requirements.txt
 ```

### Execution

Run:
```bash
python instance_segmentation.py PartB/can.jpg <your_output_folder>
```
To reproduce the result above or

```bash
python instance_segmentation.py path/to/your/image output_folder_name
```
if you want to provide your own image.
