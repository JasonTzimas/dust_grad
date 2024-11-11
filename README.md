# Dust Autograd Engine


[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/username/project/actions)

## Description
This repository implements a basic AutoGrad Engine purely in python without utilizing any 3rd party frameworks such as numpy. It works on a scalar level, using the class Value, which is of the following form:

``` python
class Value(object):
    '''
    Value Object:
        arguments:
            - x: input data of type int or float
        output:
            - Value object with value = x
    '''
    def __init__(self, x:Union[float, int, str], children=[], requires_grad=False):
        self.children = children
        self.backward_func = None
        self.grad = 0.0
        self.requires_grad = requires_grad
        self.valid_dtypes = [float, int, Value, str]
        assert type(x) in self.valid_dtypes, "Invalid input dtype {}. Should be in: {}".format(type(x), [repr(dtype) for dtype in self.valid_dtypes])
        if type(x) in [int, str]:
            try:
                x = float(x)
            except:
                raise Exception("str object cannot be converted to either float or int")
            self.val = x
        elif type(x) == Value:
            self.val = x.val
            del x
        else:
            self.val = x

    def value(self):
        return self.val
    
    def __repr__(self):
        return "Value(data={})".format(self.val)

    def __str__(self):
        return "Value(data={})".format(self.val)
    
    def __neg__(self):
        # Return a new Value instance with the negated value
        return Value(-self.val, [self])
    
    def __add__(self, other):
        if isinstance(other, type(self)):
            value = Value(self.val + other.val, [self, other])
        else:
            try: 
                value = Value(self.val + other, [self, other])
            except:
                raise Exception("{} is an invalid dtype for addition with {}".format(type(other), type(self)))
        
        value.backward_func = backward_add()
        return value

    def __radd__(self, other):
    # Call __add__ to handle the reverse addition
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, type(self)):
            value = Value(self.val - other.val, [self, other])
        else:
            try: 
                value = Value(self.val - other, [self, other])
            except:
                raise Exception("{} is an invalid dtype for addition with {}".format(type(other), type(self)))
        
        value.backward_func = backward_sub()
        return value

    def __rsub__(self, other):
    # Call __add__ to handle the reverse addition
        return self.__sub__(other)
                            
    def __mul__(self, other):
        if isinstance(other, type(self)):
            value = Value(self.val * other.val, [self, other])
            value.backward_func = backward_mul(cache=[self.val, other.val])
        else:
            try: 
                value = Value(self.val * other, [self, other])
                value.backward_func = backward_mul(cache=[self.val, other])
            except:
                raise Exception("{} is an invalid dtype for addition with {}".format(type(other), type(self)))
        return value
    
    def __rmul__(self, other):
    # Call __add__ to handle the reverse addition
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, type(self)):
            value = Value(self.val / other.val, [self, other])
            value.backward_func = backward_div(cache=[self.val, other.val])
        else:
            try: 
                value = Value(self.val / other, [self, other])
                value.backward_func = backward_div(cache=[self.val, other])
            except:
                raise Exception("{} is an invalid dtype for addition with {}".format(type(other), type(self)))
        return value

    def __rtruediv__(self, other):
    # Call __add__ to handle the reverse addition
        return self.__div__(other)
    
    def backward(self):
        self.grad = 1.0
        # -- Get topological ordering with respect to self
        topo = list(reversed(get_topological_order(self)))
        for node in topo:
            if isinstance(node, type(self)):
                if node.backward_func is not None:
                    grad = node.grad
                    grads = node.backward_func(grad)
                    for i, child in enumerate(node.children):
                        if isinstance(child, type(self)):
                            child.grad += grads[i]
                else:
                    continue

```




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
