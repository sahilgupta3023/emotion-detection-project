# Emotion Detection from Text Using BiLSTM

## Project Overview

This project demonstrates the use of a **Bidirectional Long Short-Term Memory (BiLSTM)** model for emotion detection from text. The goal is to classify text based on emotions such as **happy**, **sad**, **angry**, **surprised**, **fearful**, **disgusted**, and **neutral**.

The model is trained on a synthetic dataset and achieves a **Test Accuracy of 97.66%**.

## Files Included

1. **Emotion Detection Project Report**: `emotion_detection_project_report.pdf`
   - A detailed report describing the project, methodologies used, and results.
   
2. **Dataset**: `large_synthetic_emotion_with_neutral.csv`
   - A synthetic dataset containing text and corresponding emotions.
   
3. **Jupyter Notebook**: `emotion_detection_project.ipynb`
   - The notebook used for data preprocessing, model building, training, evaluation, and predictions.

## Libraries Used

- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation and handling the dataset.
- **Seaborn**: For visualizations (e.g., confusion matrix).
- **NeatText**: For text preprocessing such as stopword removal and special character removal.
- **Scikit-learn**: For data splitting (train-test split) and label encoding.
- **TensorFlow/Keras**: For building and training the BiLSTM model.
- **Joblib**: For saving the trained model and encoder.

## Approach

1. **Data Preprocessing**:
   - The text data was preprocessed using the **NeatText** library to remove stopwords and special characters and to convert the text to lowercase.
   
2. **Tokenization and Padding**:
   - Text data was tokenized using **Keras' Tokenizer**. The sequences were padded using **pad_sequences** to ensure uniform input size.

3. **BiLSTM Model**:
   - A **Bidirectional LSTM** model was built using **Keras** to classify the emotions in the text. The model includes an embedding layer, two BiLSTM layers, a dropout layer, and a dense output layer for multi-class classification.

4. **Model Evaluation**:
   - The model was evaluated on a test set and achieved a **Test Accuracy of 97.66%**.

## How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/emotion-detection-project.git
