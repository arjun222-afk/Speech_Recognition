"""
The Speech Emotion Recognition project aims to classify human emotions from speech using machine learning techniques.
Leveraging datasets like RAVDESS and TESS, the project processes audio features to detect emotions such as happiness, sadness, and anger.
By training models to recognize patterns in speech, the project enhances emotion-driven applications like virtual assistants and customer service.
The goal is to achieve accuracy above 75%, contributing to more empathetic human-computer interaction.
"""


import os # navigates files and directories
import librosa # for audio processing like feature extraction, loading audio files, etc.
import numpy as np # for numerical functionalities, especially with arrays
import pandas as pd # for data manipulation and analysis, creating series and dataframes
import matplotlib.pyplot as plt # for data visualization (Plots)
import seaborn as sns # for advanced data visualization (Heatmap, Confusion Matrix)
import tensorflow as tf # For building and training neural network (Feedforward NN)
from sklearn.preprocessing import LabelEncoder # for preprocessing
from sklearn.model_selection import train_test_split # for splitting the dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # for evaluation of the model
data_dir = "D:/Programming Folder/DS-JA-MM/TESS" # directory containing the TESS (Toronto Emotional Speech Set) dataset
emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "pleasant_surprise"] # list of emotions that are to be classified


# Feature Extraction
def extract_features(file_path): # takes path of the audio file
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000) # sampling rate = 16,000. determining resolution of audio signal.
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # Mel-Frequency Cepstral Coefficient -- commonly used in audio processing
        mfccs_scaled = np.mean(mfccs.T, axis=0) # feature scaling by taking the mean, dimensionality reduction (multiple data to single feature vector)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Error: {e}") # Error message if error is encountered
        return None


# Data Loading and Feature Extraction
data = [] # appened when feature vectors and their corresponding emotions are obtained


# If the emotion directory doesn't exist, it will print the error message
for emotion in emotions:
    emotion_dir = os.path.join(data_dir, emotion)
    if not os.path.exists(emotion_dir):
        print(f"Error: The directory '{emotion_dir}' does not exist.")
        continue

# For each audio file, it will extract features. When successfully extracted, the data library will be appended with
# coressponding emotion lables
    for file in os.listdir(emotion_dir):
        file_path = os.path.join(emotion_dir, file)
        print(f"Processing emotion: {emotion}, file: {file}")
        features = extract_features(file_path)
        if features is not None:
            data.append([features, emotion])

# Data Manipulation and Preparing
df = pd.DataFrame(data, columns=["features", "emotion"]) # dataframe containing 2 columns: features and emotions from the data[]
X = np.array(df["features"].tolist()) # array of features
y = np.array(df["emotion"].tolist()) # array of emotions


# Label Encoding and Splitting of Data
le = LabelEncoder() # convert categorical data to numerical data e.g. (happy, sad) -> (0,1)
y_encoded = le.fit_transform(y) # fir encoder and transform the data in one step by learing unique categories
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) # training data = 80%, testing data = 90%, random_state ensures reproducibility of data
y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=len(emotions)) # uses categorical function to make them compatible with cross-entropy loss
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=len(emotions)) # uses categorical function to make them compatible with cross-entropy loss

# Cross Entropy Loss: Commonly used loss function in classfication model, especially used in one hot encoded vector target output representing multiple classes


# Model Architecture
# Feedforward Neural Network is created by using the Tensorflow 'Sequential' API
# ReLU [Rectified Linear Unit]: Activation Fucntion that directly returns input if it's positive and 0 otherwise. Helps in introducing non-linearity
# Softmax: Activation Function that converts raw O/P Scores into probability and ensures that the sum is 1
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # input_shape defines the shape of input that this layer expects. Refers to no. of features (Columns)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(emotions), activation='softmax') # O/P neurons, One for each emotion, with softmax(used for multi-class classification)
])


# Model Compilation and Training
"""
Compiled Using Adam Optimizer, which adjusts the learning rate based on mean and variance of gradients.
Model is trained for 10 epochs with batch size of 32.
Batch Size: No. of training examples taken in one iteration, processes 32 samples and update their weights before proceeding to the next 32 samples
Epoch: No. of times the model will iterate over the entire dataset ( Here 10)
"""
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_categorical, epochs=10, batch_size=32,validation_data=(X_test, y_test_categorical))



# Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test_categorical) # evaluation is processed on the test target labels (testing set)
print(f"Test Accuracy: {accuracy * 100:.2f}%") # printing the accuracy
y_pred = model.predict(X_test) # array where each row will predict the different probabilities for different classes
y_pred_classes = np.argmax(y_pred, axis=1) #predictions are made on test set and labels are obtained using argmax() which returns the max value from the row in the y_pred array
print(classification_report(y_test, y_pred_classes, target_names=le.classes_)) # shows precision, recall and F1 Score for each emotion class

# Confusion Matrix and Data Visualization
# Confusion Matrix shows how often each class is predicted correctly or incorrectly

cm = confusion_matrix(y_test, y_pred_classes) # confusion matrix data that needs to be visualized
plt.figure(figsize=(10, 7)) # Size of the matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_) # heatmap cells are annoted with numeric decimal("d") values
plt.xlabel("Predicted") # xticklabeles: sets the labels of x-axis, typlically class name (predicted by the model)
plt.ylabel("True") # yticklabeles: sets the labels of y-axis, typlically class name (true labels of the dataset)
plt.title("Confusion Matrix") # title of the plot
plt.show() # display

# accuracy = training accuracy
# val_accuracy = testing accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# loss = training loss
# val_loss = testing loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()