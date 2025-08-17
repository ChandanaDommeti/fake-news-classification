
# 📱 Fake News Classification of SMS Spam

## 📌 Project Overview

The exponential growth of mobile communication has made SMS a major channel for spam and misinformation. This project explores **machine learning (ML)** and **deep learning (DL)** approaches to classify SMS messages as **spam (fake)** or **ham (legitimate)**.

We compare traditional ML models (with TF-IDF features) against transformer-based deep learning models (BERT, RoBERTa, DeBERTa, etc.) and lightweight custom models (TextCNN, BiLSTM+Attention) to identify the best trade-off between **accuracy, precision, and efficiency**.

---

## 🎯 Objectives

* Detect **spam/fake SMS** using ML & DL models.
* Compare **traditional ML models vs. transformer-based models**.
* Evaluate models on **accuracy, precision, recall, and F1-score**.
* Provide insights into **real-world deployment trade-offs** between lightweight and resource-intensive models.

---

## ⚙️ Methodology

### 🔹 Machine Learning Models (TF-IDF Features)

* Naive Bayes
* Random Forest
* Support Vector Classifier (SVC)
* Extra Trees
* Logistic Regression
* K-Nearest Neighbors
* XGBoost, GBDT, Bagging, AdaBoost, Decision Tree

**Best ML Performers:**

* **Naive Bayes** → Highest Precision (1.0000)
* **Extra Trees** → Highest Accuracy (0.9778)

---

### 🔹 Deep Learning Models (Transformer & Custom)

* **BERT-base**
* **RoBERTa-base**
* **DistilBERT**
* **DeBERTa-v3-small**
* **TextCNN (lightweight)**
* **BiLSTM + Attention**

**Best DL Performer:**

* **DeBERTa-v3-small** → Accuracy **0.9942** with robust precision & recall

**Lightweight Alternative:**

* **TextCNN** → \~98% accuracy with fewer resources (ideal for edge deployment)

---

## 📊 Results

| Model              | Accuracy   | Precision | Recall | F1-Score |
| ------------------ | ---------- | --------- | ------ | -------- |
| Naive Bayes        | 0.9709     | 1.0000    | -      | -        |
| Extra Trees        | 0.9778     | 0.9675    | -      | -        |
| BERT-base          | 0.9912     | 0.97      | 0.97   | 0.97     |
| RoBERTa-base       | 0.9912     | 0.97      | 0.97   | 0.97     |
| DeBERTa-v3-small   | **0.9942** | 0.97      | 0.97   | 0.97     |
| TextCNN            | 0.9797     | 0.93      | 0.92   | 0.93     |
| BiLSTM + Attention | 0.9845     | 0.92      | 0.92   | 0.92     |

---

## 🚀 Output

* **Input:** SMS text message
* **Output:**

  * **Spam (Fake):** Promotional, misleading, harmful messages
  * **Ham (Legit):** Useful, personal, safe messages
* **Confidence Score:** Indicates certainty of classification (e.g., *Spam with 99.88% confidence*)

This enables **real-time spam filtering** in SMS inboxes and messaging apps.

---

## 🔑 Key Insights

* All models achieved **>95% accuracy**.
* **Naive Bayes & SVC** → Fast, interpretable, lightweight.
* **Transformers (BERT, RoBERTa, DeBERTa)** → Robust generalization, best performance.
* **Custom models (TextCNN, BiLSTM)** → Efficient for low-resource systems.

---

## 🏁 Conclusion

* **For production environments** → DeBERTa-v3-small is best for accuracy & robustness.
* **For lightweight systems** → TextCNN provides efficiency without much performance loss.
* This project demonstrates the **practical trade-offs** between classical ML and modern DL models for **fake news detection in SMS spam**.

---


## 🛠️ Tech Stack

* **Languages:** Python
* **Libraries:** scikit-learn, TensorFlow, PyTorch, Hugging Face Transformers, XGBoost
* **Feature Extraction:** TF-IDF, Word Embeddings
* **Models:** Naive Bayes, Random Forest, SVC, Extra Trees, BERT, RoBERTa, DeBERTa, TextCNN, BiLSTM

---

## 📌 Future Work

* Deploy as an **API for SMS spam filtering**.
* Build a **real-time dashboard** for monitoring spam trends.
* Optimize models further for **mobile & edge devices**.

---

✨ **Contributors:** \[Dommeti Chandana]
📧 Contact: \[[myidchandana@gmail.com](mailto:your.email@example.com)]


