# Aircraft Image Classification with ResNet50

This repository contains a deep learning pipeline for classifying aircraft images into multiple categories using a ResNet50-based convolutional neural network. The model is trained and evaluated on an aircraft images dataset, with support for up to 10 classes. The key steps include loading and preprocessing images, fine-tuning the ResNet50 architecture, and evaluating model performance using accuracy, a confusion matrix, and a classification report.

### Key Features
- **Image Preprocessing**: 
  - Images are resized to 224x224 pixels.
  - Images are normalized by scaling pixel values to the [0, 1] range.
  
- **Model Architecture**: 
  - Based on **ResNet50**, pre-trained on the ImageNet dataset, with the top layers removed.
  - A Global Average Pooling layer and two fully connected layers are added on top of ResNet50 for fine-tuning.
  - The model uses **Adam** optimizer and is trained with **sparse categorical crossentropy** loss.

- **Training and Validation**:
  - The training data is split into 80% training and 20% validation sets.
  - The model is trained for 10 epochs with a batch size of 32.

- **Evaluation Metrics**:
  - Accuracy and loss on the test set.
  - **Confusion Matrix**: Visualized with a heatmap to show model performance across all categories.
  - **Classification Report**: Includes precision, recall, and F1-score for each class.

### Requirements
- Python 3.x
- TensorFlow/Keras
- OpenCV
- Scikit-learn
- Seaborn
- Matplotlib
- NumPy

### Usage
1. Place your aircraft images in folders under `Training/` (for training and validation) and `Test/` (for testing).
2. Each folder should represent a class (e.g., `Class_1`, `Class_2`, ..., `Class_10`).
3. Run the provided code to train the ResNet50-based model on the training data and evaluate it on the test data.
4. Visualize the results with the confusion matrix and the classification report.

### Output
- **Test Accuracy** and **Loss**
- **Confusion Matrix** for performance visualization
- **Classification Report** for detailed class-wise metrics

### License
This project is licensed under the MIT License.
