# 🐦 Twitter Sentiment Analysis

A machine-learning powered **Sentiment Analysis Web Application** built with **Python, Streamlit, and scikit-learn**. The application analyzes tweets or short text and classifies them into **Positive, Negative, Neutral, or Irrelevant** sentiment categories.

The project combines **Natural Language Processing (NLP)** with a trained machine-learning model and an interactive Streamlit interface to provide instant predictions along with a model confidence score.

---

## 🚀 Demo

> Enter any tweet or short piece of text and let the model analyze its sentiment.

### Example

```text
I absolutely love using this product! It changed my life. 🚀
```

### Output

```text
😊 Positive

Model Confidence: XX.X%
```

The exact prediction and confidence depend on the trained model and the input text.

---

## ✨ Features

* 🧠 **Machine Learning-based sentiment classification**
* 📝 Analyze tweets or arbitrary short text
* 🔤 **TF-IDF** based text feature extraction
* 🎯 Four sentiment categories:

  * 😊 Positive
  * 😞 Negative
  * 😐 Neutral
  * 🤷 Irrelevant
* 📊 Model confidence visualization
* ⚡ Instant predictions through Streamlit
* 🕓 Recent prediction history
* 🗑️ Clear prediction history
* 🚀 Pre-trained model loaded directly from serialized files
* 🎨 Custom dark-themed user interface
* 💾 Cached model loading for efficient inference

---

# 🧠 How It Works

The application follows a simple NLP inference pipeline:

```text
              User Input
                  │
                  ▼
        ┌───────────────────┐
        │    Text Input     │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │  TF-IDF Vectorizer│
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │  ML Classification│
        │       Model       │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   Label Encoder   │
        └─────────┬─────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Sentiment Prediction│
       │   + Confidence      │
       └──────────┬──────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Streamlit UI      │
        └───────────────────┘
```

### Prediction Process

1. The user enters a tweet or text.
2. The application validates the input.
3. The saved **TF-IDF vectorizer** converts the text into numerical features.
4. The trained machine-learning model predicts the sentiment class.
5. The saved **Label Encoder** converts the numerical class into a human-readable label.
6. The application calculates an estimated confidence score when supported by the model.
7. The result is displayed through the Streamlit interface.
8. The prediction is added to the recent-analysis history.

---

# 🎯 Sentiment Categories

| Sentiment         | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| 😊 **Positive**   | Text expressing positive opinions, emotions, or experiences |
| 😞 **Negative**   | Text expressing negative opinions, emotions, or experiences |
| 😐 **Neutral**    | Text without a strong positive or negative sentiment        |
| 🤷 **Irrelevant** | Text classified as irrelevant by the trained model          |

---

# 🛠️ Tech Stack

## Programming Language

* **Python**

## Machine Learning & NLP

* **scikit-learn**
* **TF-IDF Vectorization**
* Machine Learning Classification
* Label Encoding

## Web Application

* **Streamlit**

## Supporting Libraries

* **NumPy**
* **Pickle**

---

# 📁 Project Structure

```text
NLP_SENTIMENT_ANALYSIS/
│
├── app.py
│
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
│
├── twitter_training.csv
├── twitter_validation.csv
│
└── README.md
```

### File Description

| File                     | Purpose                                    |
| ------------------------ | ------------------------------------------ |
| `app.py`                 | Main Streamlit application                 |
| `sentiment_model.pkl`    | Pre-trained sentiment classification model |
| `tfidf_vectorizer.pkl`   | Fitted TF-IDF vectorizer                   |
| `label_encoder.pkl`      | Encoder used to convert class labels       |
| `twitter_training.csv`   | Training dataset                           |
| `twitter_validation.csv` | Validation dataset                         |
| `README.md`              | Project documentation                      |

---

# 🔬 Model Architecture

The project uses a traditional NLP machine-learning pipeline rather than a deep-learning or transformer-based architecture.

```text
Raw Text
   │
   ▼
TF-IDF Feature Extraction
   │
   ▼
Machine Learning Classifier
   │
   ▼
Numerical Class
   │
   ▼
Label Encoder
   │
   ▼
Sentiment Label
```

The trained components are serialized using Python's `pickle` format and loaded by the Streamlit application during runtime.

---

# 📊 Confidence Score

The application attempts to provide an estimated confidence score for each prediction.

If the trained model supports:

```python
predict_proba()
```

the highest predicted class probability is used.

If `predict_proba()` is unavailable but the model supports:

```python
decision_function()
```

the decision scores are converted into normalized scores before selecting the highest value.

The result is then displayed as a percentage in the Streamlit interface.

> **Important:** The displayed confidence should not automatically be interpreted as a perfectly calibrated probability. Model confidence depends on the underlying classifier and its calibration.

---

# 🕓 Recent Analysis History

The application maintains a temporary history of recent predictions using Streamlit session state.

Each history entry contains:

* Input text
* Predicted sentiment
* Sentiment emoji
* Confidence score

The application displays up to the **8 most recent analyses**.

Users can also clear the history using the **Clear History** button.

> History is session-based and is not stored permanently in a database.

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YASH859-hub/NLP_SENTIMENT_ANALYSIS.git
```

Move into the project directory:

```bash
cd NLP_SENTIMENT_ANALYSIS
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

# 📦 Install Dependencies

Install the required Python packages:

```bash
pip install streamlit numpy scikit-learn
```

Or create a `requirements.txt` file containing:

```text
streamlit
numpy
scikit-learn
```

Then install everything with:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 🧪 Example Predictions

### Positive

```text
I love this new phone! The camera is amazing.
```

Expected category:

```text
😊 Positive
```

---

### Negative

```text
This service was extremely disappointing.
```

Expected category:

```text
😞 Negative
```

---

### Neutral

```text
The meeting is scheduled for tomorrow at 10 AM.
```

Expected category:

```text
😐 Neutral
```

> These examples illustrate the intended use of the application. Actual predictions depend on the trained model.

---

# 📚 Dataset

The repository contains two CSV files:

### Training Dataset

```text
twitter_training.csv
```

Used as the project's training dataset.

### Validation Dataset

```text
twitter_validation.csv
```

Used for validation/evaluation during the model-development workflow.

The Streamlit application itself performs **inference using the saved model artifacts** rather than retraining the model every time the application starts.

> Before redistributing the dataset, verify its original source, license, and usage terms.

---

# ⚙️ Model Artifacts

The application depends on three serialized files.

### `sentiment_model.pkl`

Contains the trained machine-learning classification model.

### `tfidf_vectorizer.pkl`

Contains the fitted TF-IDF vectorizer used to transform text into numerical features.

### `label_encoder.pkl`

Contains the label encoder used to convert the model's numerical predictions back into sentiment labels.

All three artifacts need to remain compatible with one another.

---

# 🎨 User Interface

The application includes a custom-designed Streamlit interface featuring:

* Dark gradient background
* Centered hero section
* Text input card
* Sentiment-specific result cards
* Confidence percentage
* Confidence progress bar
* Recent analysis section
* Clear history button
* Responsive layout

The interface uses custom CSS to provide a more polished experience than the default Streamlit appearance.

---

# 🔐 Model Loading

The model artifacts are loaded using Streamlit's resource caching mechanism.

Conceptually:

```python
@st.cache_resource
def load_models():
    ...
```

This prevents the application from unnecessarily loading the model, vectorizer, and label encoder repeatedly during the same application lifecycle.

---

# ⚠️ Limitations

Although the project provides a practical NLP sentiment-analysis system, it has several limitations.

### 1. Sarcasm

Traditional TF-IDF-based models may have difficulty understanding sarcasm.

For example:

```text
Great... another software update that broke everything.
```

The literal presence of positive words can make sentiment classification challenging.

### 2. Context

The model primarily works with the text provided to it and may not understand broader conversational context.

### 3. Social Media Language

Tweets frequently contain:

* Slang
* Abbreviations
* Hashtags
* Emojis
* Mentions
* URLs
* Misspellings

These can affect prediction quality.

### 4. Confidence Calibration

The displayed confidence score should not automatically be considered a calibrated probability.

### 5. Session-Based History

Prediction history is temporary and disappears when the Streamlit session is reset.

### 6. Training Pipeline

The current repository primarily contains the trained artifacts and inference application. A separate, fully documented training pipeline would improve reproducibility.

---

# 🔮 Future Improvements

Potential improvements include:

* [ ] Add complete model-training script
* [ ] Add preprocessing pipeline
* [ ] Add model evaluation metrics
* [ ] Add confusion matrix
* [ ] Add precision, recall, and F1-score
* [ ] Compare multiple ML algorithms
* [ ] Add batch CSV prediction
* [ ] Add sentiment distribution charts
* [ ] Add prediction explainability
* [ ] Improve sarcasm detection
* [ ] Improve emoji and hashtag processing
* [ ] Add persistent prediction history
* [ ] Add REST API using FastAPI
* [ ] Add automated testing
* [ ] Add CI/CD pipeline
* [ ] Deploy the application publicly
* [ ] Experiment with transformer-based NLP models such as BERT

---

# 📈 Possible Evaluation Metrics

For a future model-training pipeline, the following metrics can be reported:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Classification Report
```

Example:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

> Add the actual values here once the final trained model has been evaluated.

---

# 🧩 Why This Project?

Sentiment analysis is a practical application of Natural Language Processing that can be used to understand opinions and emotions expressed in large collections of text.

This project demonstrates the complete transition from:

```text
Text Data
    ↓
NLP Feature Engineering
    ↓
Machine Learning
    ↓
Model Serialization
    ↓
Inference
    ↓
Interactive Web Application
```

It therefore serves as a practical demonstration of applying machine learning to a user-facing application rather than limiting the project to a notebook-based experiment.

---

# 🚀 Future Production Architecture

A more scalable version of this project could follow:

```text
                    ┌───────────────┐
                    │    Frontend   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   REST API    │
                    │   FastAPI     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ NLP Pipeline  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ ML Model      │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Prediction DB │
                    └───────────────┘
```

This architecture could allow the sentiment-analysis engine to be consumed by websites, mobile applications, dashboards, or other AI systems.

---

# 🤝 Contributing

Contributions, improvements, and suggestions are welcome.

### Development Workflow

```bash
git checkout -b feature/your-feature
```

Make your changes and test them locally.

Then:

```bash
git add .
git commit -m "feat: describe your change"
git push origin feature/your-feature
```

Finally, open a Pull Request.

---

# 📜 License

No `LICENSE` file is currently included in this repository.

If you intend to make the project open-source, consider adding an appropriate license such as the MIT License after confirming that it is suitable for the project and dataset.

---

# 👨‍💻 Author

## Yash

AI & Machine Learning Student and Developer

GitHub:

**[@YASH859-hub](https://github.com/YASH859-hub)**

Project Repository:

**[NLP_SENTIMENT_ANALYSIS](https://github.com/YASH859-hub/NLP_SENTIMENT_ANALYSIS)**

---

# ⭐ Support

If you found this project useful for learning or experimentation, consider giving the repository a ⭐ on GitHub.

---

## 📌 Project Summary

**Twitter Sentiment Analysis** is an NLP and machine-learning application that transforms textual input into sentiment predictions through a TF-IDF-based feature extraction pipeline and a pre-trained classification model.

The project demonstrates how a trained NLP model can be packaged into an interactive application using Streamlit, providing users with real-time predictions, confidence estimates, and temporary prediction history.

---

**Built with Python • NLP • Machine Learning • scikit-learn • Streamlit**
