# Phishing Email Detection Project 🚨

## Overview
This project aims to develop a phishing email detection system using machine learning. The model is built using Natural Language Processing (NLP) techniques to classify emails as either legitimate or phishing. The system uses a dataset of labeled emails, processes the text data, and trains a model to predict whether an email is a phishing attempt or not.

## Technologies Used ⚙️
- Python
- Pandas
- Scikit-learn
- TfidfVectorizer
- Multinomial Naive Bayes
- Numpy
- Google Colab

## Dataset 📊
The dataset used in this project is a collection of emails labeled as either "legitimate" or "phishing". The dataset is pre-processed to remove unnecessary characters, and the text is vectorized using `TfidfVectorizer` to extract relevant features for classification.

## Project Description 📝

1. Data Preprocessing:  
   - The dataset is loaded, and the 'text_combined' column is renamed to 'email_text'.
   - Emails are cleaned by converting the text to lowercase and removing punctuation.
   
2. Feature Extraction:  
   - The `TfidfVectorizer` is used to convert the cleaned text into numerical features that can be fed into the machine learning model.
   - A maximum of 5000 features are selected with English stop words excluded. The vectorizer also considers n-grams (unigrams and bigrams).

3. Model Training:  
   - The dataset is split into training and testing sets (80% training, 20% testing).
   - A Multinomial Naive Bayes classifier is trained on the training data.

4. Evaluation 📊:  
   - The model is tested on the test data, and accuracy is measured.
   - Additional metrics such as the classification report and confusion matrix are printed to evaluate the model’s performance.

5. Prediction Function:  
   - A function `predict_email` is defined to classify new email input as either "Phishing Email" or "Legitimate Email".

## How to Run 🚀
1. Clone the repository:
   git clone https://github.com/sharmila-ks/Phishing-Email-Detection.git
2. Install the required dependencies:
   pip install pandas scikit-learn
3. Open the .ipynb notebook in Google Colab:
   Upload the notebook file to Google Colab and run each cell interactively to see the results.
4. Test the model by running the provided code snippets inside the notebook and check for model accuracy and predictions.
- Example test:
```  print(predict_email("Urgent! Your account has been compromised. Click this link to secure it.")) ```
- Sample Output:
``` "Phishing Email" ```


## Conclusion ✅
This phishing email detection model helps identify phishing emails with high accuracy. It demonstrates the use of machine learning for detecting security threats in emails. The project can be expanded with more advanced techniques, such as deep learning, or used with larger datasets for improved performance.
