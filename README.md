# Image Classification and Time Prediction with CNNs and MLPs using TensorFlow/Keras
 This project uses TensorFlow and Keras to implement MLPs and CNNs for image classification tasks on Fashion MNIST and CIFAR-10 datasets. Additionally, a CNN model is developed for the Tell-the-Time problem to predict the time from analog clock images, experimenting with both classification and regression approaches

# Image Classification and Time Prediction with CNNs and MLPs using TensorFlow/Keras

This project implements **Multi-Layer Perceptrons (MLPs)** and **Convolutional Neural Networks (CNNs)** for image classification tasks on the **Fashion MNIST** and **CIFAR-10** datasets. Additionally, we develop a **CNN model** for the **"Tell-the-Time"** problem, predicting time from analog clock images. We explore both **classification** and **regression** approaches and experiment with label transformations.

## Overview

### Tasks:
1. **Image Classification**:
   - **Fashion MNIST**: Grayscale images representing various clothing categories.
   - **CIFAR-10**: RGB images representing 10 different classes of objects.
   
2. **Tell-the-Time Problem**:
   - **Classification Approach**: Treat the problem as a 720-class classification task, where each class corresponds to a minute on the clock.
   - **Regression Approach**: Treat the problem as a regression task to predict the time in continuous format, minimizing the "common sense" error.
   - **Multi-head Model**: A combined approach where one head predicts the hour and another predicts the minute.

### Code Structure:
1. **`fashion_mnist_mlp.py`**: MLP model for classifying **Fashion MNIST**.
2. **`fashion_mnist_cnn.py`**: CNN model for **Fashion MNIST**.
3. **`cifar10_cnn.py`**: CNN model for **CIFAR-10** dataset.
4. **`tell_the_time_cnn.py`**: CNN model for solving the **Tell-the-Time problem** with both classification and regression approaches.
5. **`common_sense_error.py`**: Function to calculate "common sense" error for time predictions.

### Neural Network Architecture:

#### **Multi-Layer Perceptron (MLP)**:
For **Fashion MNIST**, the MLP model has:
- Input: Flattened 28x28 image into a 784-dimensional vector.
- 1st Hidden Layer: 128 units with ReLU activation.
- 2nd Hidden Layer: 64 units with ReLU activation.
- Output Layer: 10 units with Softmax activation (for 10 classes).

#### **Convolutional Neural Network (CNN)**:
For **Fashion MNIST** and **CIFAR-10**, the CNN architectures consist of:
- **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation.
- **MaxPooling2D Layer**: Pooling size 2x2.
- Additional **Conv2D** and **MaxPooling2D** layers for higher complexity.
- Fully connected **Dense Layer** with 64 units.
- Output layer with 10 units (for classification).

#### **Tell-the-Time CNN Architecture**:
- **Conv2D Layers**: 32 filters, 64 filters, 128 filters with ReLU activation.
- **MaxPooling2D Layers**: Pooling size 2x2.
- **Dense Layer**: 512 units with ReLU activation.
- **Output Layer**: 72 classes (classification) or 2 continuous values (regression).

#### **Multi-head Model**:
- One head predicts hours (12 classes) and another predicts minutes (60 classes), or both as continuous regression values.

### Hyperparameters Table:

| Hyperparameter         | Values/Options                     |
|------------------------|-------------------------------------|
| **Optimizer**           | Adam (learning rate = 0.001)       |
| **Activation Function** | ReLU, Softmax                      |
| **Loss Function**       | Sparse Categorical Cross-Entropy (Classification), MSE (Regression) |
| **Batch Size**          | 32, 64, 128                        |
| **Dropout Rate**        | 0.2, 0.5                           |
| **Epochs**              | 10-100                             |
| **Learning Rate**       | 0.001, 0.0001                      |

### Metrics Table:

| Metric                 | Fashion MNIST (MLP) | Fashion MNIST (CNN) | CIFAR-10 (CNN) | Tell-the-Time (CNN) |
|------------------------|---------------------|---------------------|----------------|---------------------|
| **Training Accuracy**   | 91.01%              | 91.01%              | 94.57%         | N/A                 |
| **Validation Accuracy** | 87.82%              | 90.28%              | 89.93%         | N/A                 |
| **Test Accuracy**       | 87.24%              | 87.24%              | 87.24%         | 71.57% (Classification) |
| **Test Loss**           | 0.373               | 0.373               | 0.310          | 0.916               |

### Output:

1. **Fashion MNIST (MLP)**:
   - Training Accuracy: 91.01%, Validation Accuracy: 87.82%, Test Accuracy: 87.24%.
   - Output: Accuracy metrics and loss at each epoch.

2. **Fashion MNIST (CNN)**:
   - Training Accuracy: 91.01%, Validation Accuracy: 90.28%, Test Accuracy: 87.24%.
   - Output: Accuracy and loss metrics over 10 epochs.

3. **CIFAR-10 (CNN)**:
   - Training Accuracy: 94.57%, Test Accuracy: 89.93%.
   - Output: Accuracy and loss metrics during training and evaluation.

4. **Tell-the-Time (CNN)**:
   - Classification Test Accuracy: 71.57%.
   - Common Sense Error: (Calculated for each test prediction).
   - Output: Hour and minute predictions and accuracy/loss metrics.

### Results & Discussion:
- **Fashion MNIST**:
   - **CNN** outperforms **MLP** with a higher test accuracy (87.24% vs. 82.21%).
   - Adding more convolutional layers significantly improved performance for image classification tasks.
   
- **CIFAR-10**:
   - CNN's performance shows a major boost with accuracy reaching 89.93%.
   - Further tuning of hyperparameters like batch size and learning rate could improve results.

- **Tell-the-Time Problem**:
   - **Classification Approach**: 71.57% test accuracy; **common sense error** minimized by optimizing the architecture.
   - **Regression Approach**: Using sine and cosine transformations for labels improved accuracy.

### Conclusion:
This project showcases the implementation of **MLPs** and **CNNs** for image classification and time prediction. The **CNN** model outperformed the **MLP** in image classification tasks. The **Tell-the-Time** model demonstrated the feasibility of predicting time using images of analog clocks with a focus on minimizing the "common sense" error.

## References:
1. **TensorFlow/Keras Documentation**: https://www.tensorflow.org/
2. **Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow** by Aurelien Geron.
3. **Fashion MNIST Dataset**: https://github.com/zalandoresearch/fashion-mnist
4. **CIFAR-10 Dataset**: https://www.cs.toronto.edu/~kriz/cifar.html

