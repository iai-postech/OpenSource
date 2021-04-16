# Deep Learning-based Discriminative Refocusing of Scanning Electron Microscopy Images for Materials Science
This repository introduces the method of deep learning-based discriminative refocusing for SEM images. Our proposed model is implemented using python with jupyter notebook IDE.

# Abstract
Scanning Electron Microscopy (SEM) has contributed significantly to the development of microstructural characteristics analysis in modern-day materials science. Despite its popular usage, out-of-focus SEM images are often obtained due to improper hardware adjustments and imaging automation errors. Therefore, it is necessary to detect and restore these out-of-focus images for further analysis. Here, we propose a deep learning-based refocusing method for SEM images, particularly secondary electron (SE) images. We consider three important aspects in which are critical for an AI-based approach to be effectively applied in real-world applications: ‘Can AI refocus SEM images on non-blind settings?’, ‘Can AI refocus SEM images on blind settings?’ and ‘Can AI discriminately refocus SEM images on blind settings?’. To infer these questions, we present progressively improved approaches based on convolutional neural networks (CNN): single-scale CNN, multi-scale CNN, and multi-scale CNN powered by data augmentation, to tackle each of the above considerations, respectively. We demonstrate that our proposed method can not only refocus low-quality SEM images but can also perform the task discriminately, implying that refocusing is conducted explicitly on out-of-focused regimes within an image. We evaluate our proposed networks with martensitic SEM images in qualitative and quantitative aspects and provide further interpretations of the deep learning-based refocusing mechanism. In conclusion, our study can significantly accelerate SEM image acquisition and is applicable to data-driven platforms in materials informatics.

![Graphical Abstract](https://user-images.githubusercontent.com/36979706/114976189-86f45100-9ec0-11eb-9745-dc21f7a0a8d8.png)

# Dependencies
This project currently requires TensorFlow 1 available on Github: https://github.com/tensorflow/tensorflow.

In addition, please pip install the following packages:

* matplotlib
* numpy
* glob
* PIL
* scipy
* sklearn

# Networks

We utilize two deep neural network architectures, the single-scale CNN and the multi-scale CNN, to effectively restore deteriorated SEM images. The two networks are denoted as the single-scale refocusing network (SRN) and the multi-scale refocusing network (MRN), respectively. The SRN is fully convolutional in structure, with residual blocks served as the backbone module, as shown in Fig.


We utilized multi-scale refocusing network (MRN) to effectively refocus deteriorated SEM images. The MRN is consist of three identical single-scale refocusing network (SRN) based on convolutional neural networks (CNN). The SRN use includes the following layers in order: convolutional layer, ReLU activation function, 16 residual blocks, convolutional layer, skip-connection, and a final convolutional layer. The skip-connection directly adds the output of the initial ReLU activation function to the last convolutional layer’s input to encourage the reuse of previous features. We used cropped and augmented SEM images of size 256 x 256 as the training data of the network. Subsequently, we designed our MRN by stacking three identical SRNs using upsampling layers. The input and output sizes are halved at each subnetwork, as the networks take 256 x 256, 128 x 128, and 64 x 64 SEM images as input during the training phase. Figure below illustrates the schematic of the MRN architecture.

![architecture](https://user-images.githubusercontent.com/73891024/97993288-34017280-1e27-11eb-8bc7-46b4e6be8115.png)

# Note
Code for academic purpose only
