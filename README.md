# 📧 Spam SMS Classifier

A simple machine learning model that classifies SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and classification algorithms.

---

## 🧠 What it Does

- Cleans and processes SMS text using NLP
- Converts text to numerical features using TF-IDF
- Trains using Naive Bayes classifier
- Evaluates accuracy with confusion matrix and classification report

---

## 📁 Dataset

- Source: SMS Spam Collection Dataset
- Columns: `label` (spam/ham), `text`
- You can find the dataset in the `/dataset` folder

---

## 🚀 How to Run

```bash
git clone https://github.com/Auromirajayakumar/spam-sms-classifier.git
cd spam-sms-classifier
pip install -r requirements.txt
jupyter notebook
Then open notebook.ipynb and run all the cells.

🛠️ Tech Stack
Python 🐍

Libraries: pandas, scikit-learn, matplotlib, seaborn, nltk

Jupyter Notebook

📊 Sample Output
Add a screenshot here from your notebook (accuracy/confusion matrix)

📜 License
This project is licensed under the MIT License.

