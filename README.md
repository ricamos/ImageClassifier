# Write A Data Science Blog Post 

## Table of Contents:

1. [Motivation](#motivation)
2. [File description](#file)
3. [How to interact with your project](#interact)
4. [Licensing](#licensing)
5. [Authors](#author)
6. [Acknowledgements](#ack)

## Motivation <a name="motivation"></a>
This project is part of Udacity Data Scientist Nanodegree.

In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

In this project, you'll train an image classifier to recognize different species of flowers. 

### Part 1  - Developing an Image Classifier with Deep Learning

In this first part of the project, We work through a Jupyter notebook to implement an image classifier with PyTorch. 

### Part 2 - Building the command line application

Building the command line application that others can use. The application is a pair of Python scripts that run from the command line.

## File description <a name="file"></a>

We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below.

Image Classifier Project.ipynb - Jupyter notebook to implement an image sorter with PyTorch.

cat_to_name.jason - Is a mapping from category label to category name. This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

Train.py - train a new network on a dataset and save the model as a checkpoint.

Predict.py - uses a trained network to predict the class for an input image.

Util.py - The file with auxiliary functions for the operation of Train.py and Predict.py.

helper.py - Auxiliary file provided by Udacity

## How to interact with your project <a name="interact"></a>

* Train a new network on a data set with train.py

* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py data_directory```
  * Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  * Options:
    * Set direcotry to save checkpoints: ```python train.py data_dor --save_dir save_directory```
    * Choose arcitecture (alexnet, densenet121 or vgg16 available): ```pytnon train.py data_dir --arch "vgg16"```
    * Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20 ```
    * Use GPU for training: ```python train.py data_dir --gpu gpu```
    
* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` and return the flower name and class probability
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top **K** most likely classes:``` python predict.py input checkpoint ---top_k 3```
    * Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    * Use GPU for inference: ```python predict.py input checkpoint --gpu```

## Licensing <a name="licensing"></a>
[License file](https://github.com/ricamos/ImageClassifier/blob/master/LICENSE)

## Authors <a name="author"></a>
 Udacity provide some tips and guide me, but for the most part the code is from Ricardo Coelho.

## Acknowledgements <a name="ack"></a>
Neural Networks

[argparse module](https://docs.python.org/3/library/argparse.html)

Deep Learning With PyTorch
