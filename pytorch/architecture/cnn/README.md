# Introduction to Convolutional Neural Networks (CNNs)

Based on the paper *"An Introduction to Convolutional Neural Networks"* by Keiron O’Shea and Ryan Nash.

[CNN Overview](https://en.wikipedia.org/wiki/Convolutional_neural_network)

Convolutional Neural Networks (CNNs) are a class of deep neural networks primarily used in image recognition and classification. This project summarizes the concepts and components introduced in the referenced paper to help new learners understand the basics of CNNs.

## What Are CNNs?

Convolutional Neural Networks are inspired by the structure of the human brain, particularly the visual cortex. They are composed of multiple layers designed to process and extract features from image data in a way that captures spatial hierarchies and patterns.

CNNs are efficient at recognizing visual patterns and have become a standard architecture in computer vision tasks such as digit recognition, object detection, facial recognition, and more.

## CNN Architecture

A standard CNN architecture includes the following layers:

### 1. Input Layer

Receives image data in formats such as 64x64x3 (height, width, channels for RGB).

### 2. Convolutional Layer

Applies filters (kernels) across the input image to produce feature maps (activations). These filters are learnable parameters optimized during training.

![Convolution Operation](https://upload.wikimedia.org/wikipedia/commons/4/4f/Convolution_schematic.gif)

Each filter "slides" over the image using a technique called convolution, computing dot products between the filter and local regions of the image.

### 3. Activation Layer (ReLU)

Applies a non-linear transformation to the feature maps, introducing non-linearity into the model. Common activation functions include ReLU (Rectified Linear Unit).

### 4. Pooling Layer

Reduces the spatial dimensions (width and height) of the feature maps, keeping only the most important information and helping reduce overfitting.

![Pooling Layer](https://cs231n.github.io/assets/cnn/maxpool.jpeg)

Most commonly used pooling method: Max Pooling (e.g., 2x2 window with stride 2).

### 5. Fully Connected Layer

Each neuron is connected to every neuron in the previous layer. This layer performs classification based on the features extracted by previous layers.

### 6. Output Layer

Produces the final class probabilities using a softmax function or similar, depending on the task.

## Key Concepts

| Term                 | Description |
|----------------------|-------------|
| Receptive Field      | The region of the input image a particular neuron is responsible for. |
| Stride               | Number of pixels by which the filter slides over the input. |
| Zero Padding         | Padding added to the image borders to control output size. |
| Parameter Sharing    | Filters are shared across the image, reducing the number of parameters. |
| Overfitting          | A condition where the model learns noise and specific patterns in the training set rather than generalizing. Pooling and regularization help combat this. |

## Example: Simple CNN on MNIST

An example CNN applied to MNIST (handwritten digit classification):

Input(28 x 28)
-> Convolution Layer(5x5 filter) ->ReLU
-> Max Pooling Layer(2x2 filter)
-> Fully Connected Layer
-> Output(10 digit classes)


[MNIST](https://itp.uni-frankfurt.de/~gros/StudentProjects/WS22_23_DL_MNIST/DeepLearningMNIST.htm)

## Building a CNN: Recommended Recipe

1. Stack multiple convolutional layers followed by ReLU activations.
2. Add pooling layers to reduce dimensionality.
3. Use one or more fully connected layers for classification.
4. Prefer small filters (e.g., 3x3) and use padding to preserve dimensions.
5. Input image size should be divisible by 2 (e.g., 32x32, 64x64, 128x128).

Common layout: Conv → ReLU → Conv → ReLU → Pool → Fully Connected → Output


![CNN Recipe](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

## Optimization Strategies

- **Stride:** Use stride 1 for detailed feature extraction, larger strides to reduce dimensions.
- **Zero Padding:** Preserves spatial dimensions, preventing data shrinkage.
- **Parameter Sharing:** Reduces memory and computation.
- **Pooling Size:** Use 2x2 pooling with stride 2 for effective downsampling.

## Applications of CNNs

- Image classification (e.g., MNIST, CIFAR-10, ImageNet)
- Object detection
- Face recognition
- Medical image diagnosis
- Video classification
- Pedestrian detection in autonomous vehicles

## Authors

- **Keiron O’Shea** – Department of Computer Science, Aberystwyth University
- **Ryan Nash** – School of Computing and Communications, Lancaster University

## References

1. LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition*
2. Krizhevsky, A., Sutskever, I., Hinton, G. (2012). *ImageNet Classification with Deep CNNs*
3. Hinton, G. (2010). *A Practical Guide to Training Restricted Boltzmann Machines*
4. Zeiler, M.D., Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks*
5. Ciresan, D. et al. (2012). *Multi-column Deep Neural Networks for Image Classification*

## Conclusion

CNNs are a cornerstone of modern computer vision. By leveraging spatial hierarchies, they can extract meaningful features from images with fewer parameters than traditional ANNs. This document provides a solid foundation for understanding the architecture and operations of CNNs in a simple and practical way.
