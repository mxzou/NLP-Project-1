import sys
import os
import math
from collections import defaultdict, Counter
import re
import random
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class PerformanceTimer:
    def __init__(self):
        self.timers = defaultdict(float)
        self.starts = {}

    @contextmanager
    def timer(self, name):
        start = time.perf_counter()
        yield
        self.timers[name] += time.perf_counter() - start

    def get_stats(self):
        return dict(self.timers)

class ShakespeareanTextCategorizer:
    def __init__(self):
        self.performance = PerformanceTimer()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=50000,
            norm='l2'
        )
        self.classifiers = {
            'NaiveBayes': MultinomialNB(),
            'SVM': LinearSVC(dual=False),
            'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1)
        }
        self.best_classifier = None

    def preprocess(self, text):
        # Advanced preprocessing
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', 'NUM', text)
        return text.lower()

    def load_data(self, filename):
        data, labels = [], []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                filepath, category = line.strip().rsplit(' ', 1)
                with open(filepath, 'r', encoding='utf-8') as doc_file:
                    text = self.preprocess(doc_file.read())
                    data.append(text)
                    labels.append(category)
        return data, labels

    def train(self, train_file):
        with self.performance.timer("Training"):
            print("Loading and preprocessing data...")
            X_train, y_train = self.load_data(train_file)
            
            print("Vectorizing documents...")
            X_train_vectorized = self.vectorizer.fit_transform(X_train)
            
            print("Training and evaluating multiple classifiers...")
            best_score = 0
            for name, clf in self.classifiers.items():
                with self.performance.timer(f"Training {name}"):
                    scores = cross_val_score(clf, X_train_vectorized, y_train, cv=5)
                    avg_score = np.mean(scores)
                    print(f"{name} Average Score: {avg_score:.4f}")
                    if avg_score > best_score:
                        best_score = avg_score
                        self.best_classifier = clf

            print(f"Best classifier: {type(self.best_classifier).__name__}")
            self.best_classifier.fit(X_train_vectorized, y_train)

    def predict(self, test_file):
        with self.performance.timer("Prediction"):
            X_test, _ = self.load_data(test_file)
            X_test_vectorized = self.vectorizer.transform(X_test)
            predictions = self.best_classifier.predict(X_test_vectorized)
            return predictions

    def cross_validate(self, train_file, k=5):
        with self.performance.timer("Cross-validation"):
            X, y = self.load_data(train_file)
            X_vectorized = self.vectorizer.fit_transform(X)
            scores = cross_val_score(self.best_classifier, X_vectorized, y, cv=k)
            return scores

def log_performance(stats, accuracies=None):
    print("\n=== Performance Metrics ===")
    for key, value in stats.items():
        print(f"{key}: {value:.4f} seconds")
    if accuracies is not None:
        print(f"Accuracies: {', '.join(f'{acc:.4f}' for acc in accuracies)}")
        print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print("===========================\n")

def main():
    if len(sys.argv) != 4:
        print("Usage: python shakespearean_text_categorizer.py <mode> <train_file> <test_file>")
        sys.exit(1)

    mode, train_file, test_file = sys.argv[1:4]
    categorizer = ShakespeareanTextCategorizer()

    print(f"Initializing Shakespearean NLP Text Categorizer")
    print(f"Mode: {mode}")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")

    if mode == "train_test":
        print("\nCommencing training phase...")
        categorizer.train(train_file)

        print("\nInitiating prediction phase...")
        predictions = categorizer.predict(test_file)

        with open("predictions.labels", 'w', encoding='utf-8') as f_out:
            with open(test_file, 'r', encoding='utf-8') as f_in:
                for prediction, line in zip(predictions, f_in):
                    filepath = line.strip()
                    f_out.write(f"{filepath} {prediction}\n")

        print(f"Prediction complete. Results written to predictions.labels")
        log_performance(categorizer.performance.get_stats())

    elif mode == "cross_validate":
        print("\nInitiating cross-validation...")
        accuracies = categorizer.cross_validate(train_file)
        log_performance(categorizer.performance.get_stats(), accuracies)

    else:
        print("Invalid mode. Use 'train_test' or 'cross_validate'.")
        sys.exit(1)

    print("Shakespearean NLP Text Categorizer execution complete.")

if __name__ == "__main__":
    main()