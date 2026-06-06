# Spam Email Detection System

## Project Overview

This project is a Machine Learning-based Spam Email Detection System that classifies messages as Spam or Ham (Not Spam). The model is trained on a real-world dataset using the Multinomial Naive Bayes algorithm.

## Objective

The objective of this project is to automatically identify spam messages using Natural Language Processing (NLP) and Machine Learning techniques.

## Features

- Spam and Ham message classification
- Real-world dataset from Kaggle
- Text vectorization using CountVectorizer
- Multinomial Naive Bayes model
- Accuracy evaluation
- Confusion Matrix
- User message prediction

## Technologies Used

- Python
- Pandas
- Scikit-learn

## Dataset

Dataset Source: Kaggle SMS Spam Collection Dataset

- Total Messages: 5572
- Classes:
  - Spam
  - Ham

## Skills Demonstrated

- Python Programming
- Data Preprocessing
- Machine Learning
- Text Classification
- Feature Extraction
- Model Evaluation

## Algorithm Used

### CountVectorizer
Converts text messages into numerical vectors.

### Multinomial Naive Bayes
Classifies messages as Spam or Ham based on learned patterns.

## Project Workflow

1. Load dataset from CSV file
2. Preprocess data
3. Split data into training and testing sets
4. Convert text into numerical vectors
5. Train Naive Bayes model
6. Evaluate model performance
7. Predict user-entered messages

## Repository structure
spam-email-detection/
│
├── .gitignore
├── spam_detector.py
├── spam.csv
└── README.md

## Installation

```bash
pip install pandas scikit-learn
```

## Run the Project

```bash
python spam_detector.py
```

## Sample Output

```text
Accuracy: 98.5%

Prediction: SPAM
Warning: This email is likely SPAM.
```

## Output Screenshot

<img width="885" height="356" alt="image" src="https://github.com/user-attachments/assets/405e20c2-fc9f-442f-8366-92abde8f932e" />
<img width="907" height="294" alt="image" src="https://github.com/user-attachments/assets/ee4fcd6c-b223-4d2c-9060-66505c6bf246" />


## Future Improvements

- Use advanced NLP techniques
- Compare multiple machine learning models
- Build a GUI application
- Deploy as a web application

## Conclusion

This project demonstrates how Machine Learning and NLP can be used to detect spam messages effectively using a real-world dataset.

## Author

Tamanna Verma
