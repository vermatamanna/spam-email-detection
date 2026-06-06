# Spam Email Detection

## Project Overview
This project is a simple Spam Email Detection System built using Python and Machine Learning. It classifies email messages as Spam or Ham (Not Spam) using the Multinomial Naive Bayes algorithm.

## Features
- Spam/Ham classification
- CountVectorizer for text processing
- Multinomial Naive Bayes model
- Accuracy calculation
- Confusion Matrix
- User input prediction

## Technologies Used
- Python
- Pandas
- Scikit-learn

## Algorithm Used
- Multinomial Naive Bayes

## Project Workflow
1. Create dataset
2. Convert dataset into DataFrame
3. Split dataset into training and testing data
4. Convert text into vectors
5. Train the model
6. Make predictions
7. Calculate accuracy
8. Display confusion matrix
9. Predict user-entered email

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python spam_detector.py
```

===== MODEL EVALUATION =====
Actual Labels   : ['spam', 'ham']
Predicted Labels: ['spam', 'ham']

Accuracy: 100.0 %

Confusion Matrix:
[[1 0]
 [0 1]]

===== SPAM EMAIL CHECKER =====
Enter an email message:

<img width="397" height="164" alt="image" src="https://github.com/user-attachments/assets/7770a7b8-1c21-4631-b59a-ee83156327e0" />


## Future Improvements
- Use a larger dataset
- Add GUI support
- Deploy as a web application

## Conclusion
This project demonstrates how Machine Learning can be used to automatically detect spam emails.

## Mini Porject
submitted by - Tamanna Verma
