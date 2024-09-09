# Object Detection with Real-Time Webcam Classification

This repository contains two Jupyter notebooks that implement a real-time object classification system using a trained model and webcam input.
1. TrainedModel.ipynb: This notebook focuses on image processing, model building, and training. It preprocesses the dataset, builds a machine learning model, trains it, and saves the trained model for later use.
2. WebCam_App.ipynb: This notebook utilizes the trained model for real-time object classification from a webcam feed. It loads the model and performs real-time classification on the video stream.

### Dataset Description
The dataset is downloaded online.

### Image Preprocessing
Includes techniques like resizing, normalization, data augmentation and data generation to prepare the dataset for model training.
  1. Rescaling: Images are resized to a uniform size (512x384) for consistency.
  2. Normalization: Pixel values are scaled to the range [0, 1] for faster model convergence.
  3. Data augmentation to prevent overfitting and improve the model's generalizability through Horizontal/Vertical Flip and Shear/Zoom/Shift
  6. train_generator, val_generator, and test_generator efficiently load and preprocess data for training, validation, and testing.
  7. The Class labels dictionary maps class indices to their corresponding names for interpreting predictions.
 
### Model Architecture (Inception-based CNN)
By utilizing an Inception-based Convolutional Neural Network (CNN) architecture for object classification, the model consists of:
  1. Input Layer: Takes an image of size 512x384x3 as input.
  2. Convolutional Layers: Employs multiple convolutional layers with 3x3 kernels and ReLU activation for feature extraction.
  3. Max Pooling Layers: Reduces the dimensionality of feature maps through max pooling with a pool size of 2x2 and stride of 2x2.
  4. Inception Blocks: Incorporates Inception blocks, which consist of parallel convolutional filters with different kernel sizes, to capture diverse features at multiple scales.
  5. Average Pooling Layer: Performs average pooling to summarize feature maps.
  6. Flatten Layer: Converts the feature maps into a single vector.
  7. Dropout Layer: Introduces randomness to prevent overfitting.
  8. Dense Layer: Outputs the predicted class probabilities using a softmax activation function.
  <img width="874" alt="Screenshot 2024-09-09 at 15 37 34" src="https://github.com/user-attachments/assets/21e3c0b9-9306-41f8-b324-bb3e387fd798">
 
### Model Training
The model is trained for 50 epochs using the training and validation data generators.
  1. The train_generator provides batches of training data to the model.
  2. The validation_data and validation_steps parameters are used to evaluate the model's performance on the validation set during training.
  3. The steps_per_epoch parameter specifies the number of training steps per epoch, ensuring that all training data is used for each epoch.
     
### Real-Time Classification
Demonstrates how the integration of a trained model with a webcam for real-time object classification and enables seamless communication between the frontend and backend. This creates a real-time object classification experience where the user can see the predicted object label on the screen as they interact with the webcam.
  1. The script continuously captures photos from the webcam and sends them to the Predict function.
  2. The Predict function processes the images, predicts the object class, and displays the results in the HTML output.

### Performance Evaluation:
The results demonstrate that the object classification model achieves reasonable accuracy and generalizability. However, there is room for improvement in reducing the overfitting and improving the model's performance on unseen data.
<img width="764" alt="Screenshot 2024-09-09 at 16 26 07" src="https://github.com/user-attachments/assets/4876b3a8-1e5c-492a-9352-e8c44b25758e">

The model is particularly good at classifying class 3, with 100% accuracy. However, it is not as good at classifying class 0, with only 50% accuracy. This may indicate that the model being overfit to the training data.
<img width="477" alt="Screenshot 2024-09-09 at 16 28 35" src="https://github.com/user-attachments/assets/636fd0cf-64da-4227-99f1-59de33e95614">

## Getting Started:
1. Install Requirements: Ensure you have Python and the necessary libraries (OpenCV, scikit-image, TensorFlow/PyTorch) installed.
2. Download Dataset: Download the dataset used for training the model. 
3. Run Notebooks: Open and run the Jupyter notebooks in the following order:
    TrainedModel.ipynb
    WebCam_App.ipynb
   




