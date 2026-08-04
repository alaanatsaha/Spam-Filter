# Spam Filter Chat Bot

## Overview

Spam Filter Chat Bot is a lightweight machine learning web application designed to classify Arabic text messages into two categories: **Spam (مزعج)** and **Important (مهم)**.

The project uses a Flask backend with a chatbot-style user interface, allowing users to enter Arabic messages and receive instant classification results.

---

## Features

- Arabic chatbot-style interface with RTL support
- Real-time spam message classification
- Text classification using TF-IDF and Multinomial Naive Bayes
- Flask REST API with `/predict` endpoint
- Simple and interactive user experience
- Reset conversation functionality

---

## Technologies Used

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Pandas

### Frontend
- HTML
- CSS
- JavaScript

---

## How It Works

1. The application loads and trains a machine learning model using a labeled Arabic dataset.
2. Text messages are converted into numerical features using TF-IDF Vectorization.
3. The Multinomial Naive Bayes algorithm analyzes the message content.
4. The user enters a message through the chatbot interface.
5. The message is sent to the Flask `/predict` API endpoint.
6. The model returns a prediction:
   - Spam (مزعج)
   - Important (مهم)
7. The result is displayed instantly in the chat interface.

---

## Project Structure
