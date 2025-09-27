import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier # Changed: Import MLPClassifier from scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- NLTK Downloads (Ensuring necessary data is available) ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Data Loading and Splitting ---
# This section comes BEFORE any text preprocessing or TF-IDF.

# 1. Load your dataset
try:
    df = pd.read_csv('enron_spam_data.csv')
    print("Dataset loaded successfully.")
    print(f"Original dataset shape: {df.shape}")
    print("Actual columns:", df.columns.tolist())  # Debug: Show actual columns
except FileNotFoundError:
    print("Error: 'enron_spam_data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# 2. Separate Features (X) and Labels (y)
X = df['Message']  # Fixed: Use 'Message' instead of 'text'
y_raw = df['Spam/Ham']  # Fixed: Use 'Spam/Ham' instead of 'label'

# Convert 'ham'/'spam' strings to numerical 0/1 for model training.
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Labels encoded: Original classes {le.classes_} mapped to numerical values.")

# 3. Split the Data into Training and Testing Sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n--- Data Split Summary ---")
print(f"Original dataset size: {len(df)} emails")
print(f"Training set size (raw text): {len(X_train_raw)} emails")
print(f"Test set size (raw text): {len(X_test_raw)} emails")

print("\nDistribution of numerical labels in original dataset:")
print(pd.Series(y).value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

print("\nDistribution of numerical labels in training set:")
print(pd.Series(y_train).value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

print("\nDistribution of numerical labels in test set:")
print(pd.Series(y_test).value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))


# --- Data Preprocessing ---
stop_words = set(stopwords.words('english')) # [cite: 40]
porter = PorterStemmer() # [cite: 42]

def preprocess_text(text):
    if not isinstance(text, str): # Robustness check
        return ""
    text = text.lower() # Lowercasing [cite: 39]
    text = re.sub(r'[^a-z\s]', '', text) # Special Character Removal [cite: 43]
    tokens = nltk.word_tokenize(text) # Tokenization [cite: 41]
    tokens = [word for word in tokens if word not in stop_words] # Stopword Removal [cite: 40]
    tokens = [porter.stem(word) for word in tokens] # Stemming [cite: 42]
    return ' '.join(tokens)

print("\n--- Applying Data Preprocessing ---")
X_train_processed = X_train_raw.apply(preprocess_text)
X_test_processed = X_test_raw.apply(preprocess_text)
print("Preprocessing complete for training and test sets.")

# --- Feature Extraction (TF-IDF) ---
vectorizer = TfidfVectorizer(max_features=5000) # TF-IDF [cite: 45]

print("\n--- Performing TF-IDF Feature Extraction ---")
X_train_tfidf = vectorizer.fit_transform(X_train_processed)
X_test_tfidf = vectorizer.transform(X_test_processed)
print("TF-IDF feature extraction complete.")
print(f"Shape of TF-IDF features for training: {X_train_tfidf.shape}")
print(f"Shape of TF-IDF features for testing: {X_test_tfidf.shape}")


# --- Model Training: Naïve Bayes Classifier ---
nb_model = MultinomialNB() # Naïve Bayes Classifier [cite: 49]
nb_model.fit(X_train_tfidf, y_train) # [cite: 59]
print("\nNaïve Bayes model trained.")

nb_train_probs = nb_model.predict_proba(X_train_tfidf) # Get probabilities [cite: 47]
nb_test_probs = nb_model.predict_proba(X_test_tfidf) # Get probabilities [cite: 47]

spam_class_idx = list(le.classes_).index('spam') # Dynamically get index of 'spam'
nb_train_spam_prob = nb_train_probs[:, spam_class_idx].reshape(-1, 1) # [cite: 60]
nb_test_spam_prob = nb_test_probs[:, spam_class_idx].reshape(-1, 1) # [cite: 60]


# --- Combine TF-IDF features with Naïve Bayes probabilities for MLP ---
# Convert sparse TF-IDF matrices to dense arrays before hstack
X_train_hybrid = np.hstack((X_train_tfidf.toarray(), nb_train_spam_prob)) # [cite: 53]
X_test_hybrid = np.hstack((X_test_tfidf.toarray(), nb_test_spam_prob)) # [cite: 53]

print(f"Shape of hybrid features for training (TF-IDF + NB Prob): {X_train_hybrid.shape}")
print(f"Shape of hybrid features for testing (TF-IDF + NB Prob): {X_test_hybrid.shape}")

# --- Model Training: Multi-Layer Perceptron (MLP) using Scikit-learn ---
# Replaced TensorFlow Sequential model with Scikit-learn's MLPClassifier
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128,),  # One hidden layer with 128 neurons
    activation='relu',
    solver='adam',
    max_iter=500,               # Increased max_iter (e.g., 500, adjust as needed)
    random_state=42,            # For reproducibility
    early_stopping=True,        # Optional: to stop training if validation score isn't improving
    n_iter_no_change=10,        # Optional: number of iterations with no improvement to wait before early stopping
    verbose=True                # To see training progress and convergence
    # For more advanced neural network features like Dropout, deep learning libraries
)

print("\n--- Training MLP Model (Scikit-learn) ---")
# Train the MLP model [cite: 61]
mlp_model.fit(X_train_hybrid, y_train)
print("MLP model trained.")

# --- Evaluation ---
print("\n--- Model Evaluation ---")

# Evaluate Naïve Bayes Model
y_pred_nb = nb_model.predict(X_test_tfidf)
print("\n--- Naïve Bayes Model Performance ---")
print(f"Accuracy (NB): {accuracy_score(y_test, y_pred_nb):.4f}") # [cite: 63]
print(f"Precision (NB): {precision_score(y_test, y_pred_nb):.4f}") # [cite: 64]
print(f"Recall (NB): {recall_score(y_test, y_pred_nb):.4f}") # [cite: 65]
print(f"F1-Score (NB): {f1_score(y_test, y_pred_nb):.4f}") # [cite: 66]

# Confusion Matrix for Naïve Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix for Naïve Bayes Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_naive_bayes.png')
plt.show()

# Evaluate MLP Model (Scikit-learn)
# MLPClassifier's predict() directly returns binary predictions, no thresholding needed
y_pred_mlp = mlp_model.predict(X_test_hybrid)

print("\n--- Hybrid Naïve Bayes + Scikit-learn MLP Model Performance ---")
print(f"Accuracy (Scikit-learn MLP): {accuracy_score(y_test, y_pred_mlp):.4f}")
print(f"Precision (Scikit-learn MLP): {precision_score(y_test, y_pred_mlp, zero_division=0):.4f}")
print(f"Recall (Scikit-learn MLP): {recall_score(y_test, y_pred_mlp, zero_division=0):.4f}")
print(f"F1-Score (Scikit-learn MLP): {f1_score(y_test, y_pred_mlp, zero_division=0):.4f}")

# Confusion Matrix for Scikit-learn MLP
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix for Hybrid Naïve Bayes + Scikit-learn MLP')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_hybrid_mlp.png')
plt.show()

# End of script