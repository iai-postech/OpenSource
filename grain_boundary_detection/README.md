# Label-free Grain Segmentation for Optical Microscopy Images via Unsupervised Image-to-Image Translation
This repository introduces the novel label-free grain boundary detection in the field of optical microscopy. Our proposed method is implemented using python with jupyter notebook IDE.

# Abstract
Grain boundaries play an important role in governing the mechanical and physical properties of polycrystalline materials. Therefore, quantitative analysis of grain structure is an important prerequisite for establishing structure-property relationships. However, this quantitative analysis is hampered by the low contrast and non-uniform illumination of optical microscopy images. Although previous studies have used neural networks to detect grain boundaries in optical micrographs, their practical applications have been restricted due to labeling costs for supervised learning and their sensitivity to defects present in a grain structure image. In this paper, we approach grain boundary detection as a real-to-virtual (R2V) translation problem, mapping a single real microstructure to virtual microstructures. With this perspective, our framework benefits from two learning schemes: unsupervised learning and physics-inspired learning. Extensive experiments demonstrate the superiority and generality of our framework. We expect that our approach will facilitate the use of data-driven techniques in the field of quantitative microstructure analysis.

<img width="80%" src="https://ifh.cc/g/4FQKrA.png"/>

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
