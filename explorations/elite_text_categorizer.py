import sys
import os
import math
from collections import defaultdict, Counter
import re
import random
import time
import json
from contextlib import contextmanager

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

class TextCategorizer:
    def __init__(self):
        self.categories = set()
        self.vocab = set()
        self.cat_doc_counts = Counter()
        self.cat_word_counts = defaultdict(Counter)
        self.total_docs = 0
        self.word_counts = Counter()
        self.category_priors = {}
        self.performance = PerformanceTimer()

    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def train(self, train_data):
        with self.performance.timer("Training"):
            for filepath, category in train_data:
                self.categories.add(category)
                self.cat_doc_counts[category] += 1
                self.total_docs += 1
                
                with open(filepath, 'r', encoding='utf-8') as doc_file:
                    tokens = self.tokenize(doc_file.read())
                    unique_tokens = set(tokens)
                    self.vocab.update(unique_tokens)
                    self.cat_word_counts[category].update(unique_tokens)
                    self.word_counts.update(unique_tokens)

            for category in self.categories:
                self.category_priors[category] = math.log(self.cat_doc_counts[category] / self.total_docs)

    def predict(self, filepath):
        with self.performance.timer("Prediction"):
            with open(filepath, 'r', encoding='utf-8') as f:
                tokens = set(self.tokenize(f.read()))
            
            scores = {category: self.category_priors[category] for category in self.categories}
            vocab_size = len(self.vocab)
            
            for category in self.categories:
                cat_word_total = sum(self.cat_word_counts[category].values())
                for token in tokens & self.vocab:
                    count = self.cat_word_counts[category][token]
                    scores[category] += math.log((count + 1) / (cat_word_total + vocab_size))
            
            return max(scores, key=scores.get)

    def categorize(self, test_data):
        with self.performance.timer("Categorization"):
            return [(filepath, self.predict(filepath)) for filepath, _ in test_data]

    def cross_validate(self, all_data, k=5):
        with self.performance.timer("Cross-validation"):
            random.shuffle(all_data)
            fold_size = len(all_data) // k
            accuracies = []

            for i in range(k):
                start, end = i * fold_size, (i + 1) * fold_size
                test_data = all_data[start:end]
                train_data = all_data[:start] + all_data[end:]

                self.__init__()
                self.train(train_data)
                predictions = self.categorize(test_data)
                
                accuracy = sum(pred == true for (_, pred), (_, true) in zip(predictions, test_data)) / len(test_data)
                accuracies.append(accuracy)

            return accuracies

    def get_performance_stats(self):
        return self.performance.get_stats()

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip().rsplit(' ', 1) for line in f]

def log_performance(stats, accuracies=None):
    print("\n=== Performance Metrics ===")
    for key, value in stats.items():
        print(f"{key}: {value:.4f} seconds")
    if accuracies:
        print(f"Accuracies: {', '.join(f'{acc:.4f}' for acc in accuracies)}")
        print(f"Average Accuracy: {sum(accuracies) / len(accuracies):.4f}")
    print("===========================\n")

def main():
    if len(sys.argv) != 4:
        print("Usage: python text_categorizer.py <mode> <train_file> <test_file>")
        sys.exit(1)

    mode, train_file, test_file = sys.argv[1:4]
    categorizer = TextCategorizer()

    print(f"Initializing Elite NLP Text Categorizer")
    print(f"Mode: {mode}")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")

    if mode == "train_test":
        print("\nCommencing training phase...")
        train_data = load_data(train_file)
        categorizer.train(train_data)
        print(f"Training complete. Processed {categorizer.total_docs} documents.")
        print(f"Vocabulary size: {len(categorizer.vocab)} unique terms")
        print(f"Categories detected: {', '.join(categorizer.categories)}")

        print("\nInitiating prediction phase...")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = [(line.strip(), None) for line in f]

        predictions = categorizer.categorize(test_data)

        with open("predictions.labels", 'w', encoding='utf-8') as f_out:
            for filepath, category in predictions:
                f_out.write(f"{filepath} {category}\n")

        print(f"Prediction complete. Results written to predictions.labels")
        log_performance(categorizer.get_performance_stats())

    elif mode == "cross_validate":
        print("\nInitiating cross-validation...")
        all_data = load_data(train_file)
        accuracies = categorizer.cross_validate(all_data)
        log_performance(categorizer.get_performance_stats(), accuracies)

    else:
        print("Invalid mode. Use 'train_test' or 'cross_validate'.")
        sys.exit(1)

    print("Elite NLP Text Categorizer execution complete.")

if __name__ == "__main__":
    main()