# Image Segmentation using U-Net in TensorFlow

## Overview
This project implements an image segmentation model using the U-Net architecture with TensorFlow and Keras. The model is trained on a custom dataset consisting of images and their corresponding segmentation masks. The dataset is preprocessed by dividing images into patches and normalizing them for better training efficiency.

## Features
- Uses U-Net architecture with an encoder-decoder structure.
- Image patchification for efficient memory usage.
- Data normalization using MinMaxScaler.
- Support for multi-class semantic segmentation.
- Implements early stopping for better generalization.
- Visualizes training history and predictions.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn pillow
```


## Dataset Structure
The dataset should be organized as follows:
```
root_dir/
│── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
│── masks/
│   ├── mask1.png
│   ├── mask2.png
│   ├── ...
```

## How to Use
### 1. Mount Google Drive (if using Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Verify GPU Availability
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)
```

### 3. Preprocess Data
- Load images and masks.
- Crop images to be divisible by patch size (256x256).
- Normalize image patches.
- Convert mask values to categorical labels.

### 4. Train U-Net Model
```python
model.fit(
    batch_generator(X_train, y_train_cat, BATCH_SIZE),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    verbose=1
)
```

### 5. Visualize Predictions
```python

import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()
```


## Model Architecture
The U-Net model consists of:
- **Encoder:** Convolutional layers with ReLU activation followed by max-pooling.
- **Bridge:** Convolutional layers with dropout.
- **Decoder:** Transposed convolutions with concatenation from the encoder layers.
- **Output Layer:** Softmax activation for multi-class classification.

## Evaluation Metrics
The model is evaluated based on:
- **Categorical Cross-Entropy Loss**: Measures how well the predicted segmentation matches the ground truth.
- **Accuracy**: The proportion of correctly classified pixels.

## Results
- The training and validation accuracy is plotted using `matplotlib`.
- Segmentation predictions are visualized alongside the original images.
<img width="685" alt="Screenshot 2025-02-26 at 00 22 07" src="https://github.com/user-attachments/assets/ff5e0882-6d73-419c-9e23-7537ce3f6dff" />


## Future Improvements
- Experiment with different backbone architectures (ResNet, EfficientNet, etc.).
- Implement data augmentation techniques for better generalization.
- Optimize hyperparameters using automated tuning.
- Deploy the model as a web application for real-time segmentation.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
- Inspired by the original U-Net paper: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- TensorFlow and Keras documentation.



