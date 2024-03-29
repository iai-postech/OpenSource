# A Unified Microstructure Segmentation Approach via Human-In-The-Loop Machine Learning
This repository introduces the weakly supervised learning approach for microstructure segmentation. Our proposed method is implemented using python with jupyter notebook IDE.

# Abstract
Microstructure segmentation, which is used to extract statistical description of microstructures, is an essential step for establishing quantitative structure-property relationships in a broad range of materials research areas. The task is challenging due to the morphological complexity and diversity of structural constituents, as well as the low-contrast and non-uniform illumination nature of microscopic imaging systems. While recent breakthroughs in deep learning have led to significant progress in microstructure segmentation, there remain two fundamental challenges: prohibitive annotation costs and the black-box nature of deep learning. To tackle these challenges, we propose a human-in-the-loop machine learning framework for unified microstructure segmentation, which leverages recent advances in both weakly supervised learning and active learning techniques. Our key insight is the integration of human and machine capabilities to achieve accurate and reliable microstructure segmentation at minimal annotation costs. Extensive quantitative and qualitative experiments demonstrate the generality of our framework across material classes, structural constituents, and microscopic imaging modalities.

<img width="80%" src="https://user-images.githubusercontent.com/36979706/213759954-9cfd671c-d9f7-4070-8371-80a2394df863.jpg"/>

# Dependencies
This project currently requires the following packages:

* torch 1.12.0
* matplotlib 3.5.2
* numpy 1.23.1
* glob
* Pillow 9.2.0
* opencv-python 4.6.0.66 
* skimage 0.19.3

# Networks
We utilize fully convolutional networks as the backbone model, which extracts the feature vector from each pixel and then classifies it into one of several semantic classes. The networks have the following layers in order: four convolutional modules, a $1 \times 1$ convolutional layer, and a batch normalization layer. Each convolutional module consists of a $3 \times 3$ convolutional layer, rectified linear unit (ReLU) activation function, and a batch normalization layer.

# Note
Code for academic purpose only
