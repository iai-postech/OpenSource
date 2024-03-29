# Label-free Grain Segmentation for Optical Microscopy Images via Unsupervised Image-to-Image Translation
This repository introduces the novel label-free grain boundary detection in the field of optical microscopy. Our proposed method is implemented using python with jupyter notebook IDE.

# Abstract
Grain boundaries play an important role in governing the mechanical and physical properties of polycrystalline materials. Therefore, quantitative analysis of grain structure is an important prerequisite for establishing structure-property relationships. However, this quantitative analysis is hampered by the low contrast and non-uniform illumination of optical microscopy images. Although previous studies have used neural networks to detect grain boundaries in optical micrographs, their practical applications have been restricted due to labeling costs for supervised learning and their sensitivity to defects present in a grain structure image. In this paper, we approach grain boundary detection as a real-to-virtual (R2V) translation problem, mapping a single real microstructure to virtual microstructures. With this perspective, our framework benefits from two learning schemes: unsupervised learning and physics-inspired learning. Extensive experiments demonstrate the superiority and generality of our framework. We expect that our approach will facilitate the use of data-driven techniques in the field of quantitative microstructure analysis.

<img width="80%" src="https://ifh.cc/g/hChRsW.jpg"/>

# Dependencies
This project currently requires the following packages:

* torch 1.10.1+cu102
* matplotlib 3.5.2
* numpy 1.20.3
* glob
* Pillow 9.1.1
* opencv-python 4.1.2.30
* skimage 0.19.2
* albumentations 1.1.0

# Networks
We utilize Cycle-consistent Generative Adversarial Networks (CycleGAN) for the backbone of our networks.

# Scripts

Please write with own your directory in Path.json file and uploade your optical microscopy image in data_files/trainA. There are two types of images: black and white virtual images.
If you want to generate white clean image, please write white_normal in GrainboundaryDetection.ipynb.
If you want to generate black clean image, please write black_normal in GrainboundaryDetection.ipynb.

# Note
Code for academic purpose only
