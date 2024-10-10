import sys
import os
import math
from collections import defaultdict, Counter
import re
import random
import time
from contextlib import contextmanager
import heapq

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

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = defaultdict(int)

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, category):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count[category] += 1
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return defaultdict(int)
            node = node.children[char]
        return node.count if node.is_end else defaultdict(int)

class TextCategorizer:
    def __init__(self):
        self.categories = set()
        self.trie = Trie()
        self.cat_doc_counts = Counter()
        self.total_docs = 0
        self.performance = PerformanceTimer()
        self.stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'but'])
        self.feature_scores = Counter()
        self.top_features = set()
        self.category_priors = {}
        self.category_word_counts = {}  # Add this line

    def preprocess(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', 'NUM', text)
        tokens = text.lower().split()
        return [self.stem(token) for token in tokens if token not in self.stop_words]

    def stem(self, word):
        if len(word) > 3:
            if word.endswith('ing'):
                return word[:-3]
            elif word.endswith('ly'):
                return word[:-2]
            elif word.endswith('ies'):
                return word[:-3] + 'y'
            elif word.endswith('es'):
                return word[:-2]
            elif word.endswith('s') and not word.endswith('ss'):
                return word[:-1]
        return word

    def get_ngrams(self, tokens, n=3):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def train(self, train_data):
        if isinstance(train_data, str):
            # It's a file path, open and read the file
            with open(train_data, 'r', encoding='utf-8') as f:
                # Process file contents
                pass
        elif isinstance(train_data, list):
            # It's already a list of data, process directly
            for line in train_data:
                # Process each line of data
                pass
        else:
            raise TypeError("train_data must be a file path (str) or a list of data")

        # ... rest of the training logic ...

    def predict(self, filepath):
        with self.performance.timer("Prediction"):
            with open(filepath, 'r', encoding='utf-8') as f:
                tokens = self.preprocess(f.read())
                ngrams = self.get_ngrams(tokens)
            
            scores = {}
            for category, word_counts in self.category_word_counts.items():
                score = 0
                for word, count in document_words.items():
                    if word in word_counts:
                        score += count * word_counts[word]
                scores[category] = score

            if not scores:
                return "unknown"  # or some default category
            
            return max(scores, key=scores.get)

    def categorize(self, test_file, output_file):
        with self.performance.timer("Categorization"):
            with open(test_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    filepath = line.strip()
                    category = self.predict(filepath)
                    f_out.write(f"{filepath} {category}\n")

    def cross_validate(self, train_file, k=5):
        with self.performance.timer("Cross-validation"):
            with open(train_file, 'r', encoding='utf-8') as f:
                all_data = [line.strip().rsplit(' ', 1) for line in f]
            
            random.shuffle(all_data)
            fold_size = len(all_data) // k
            accuracies = []

            for i in range(k):
                start = i * fold_size
                end = start + fold_size if i < k - 1 else len(all_data)
                test_data = all_data[start:end]
                train_data = all_data[:start] + all_data[end:]

                self.__init__()
                self.train(train_data)

                correct = sum(1 for filepath, true_category in test_data if self.predict(filepath) == true_category)
                accuracy = correct / len(test_data)
                accuracies.append(accuracy)

            return accuracies

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
        print("Usage: python nlp_categorizer_masterpiece.py <mode> <train_file> <test_file>")
        sys.exit(1)

    mode, train_file, test_file = sys.argv[1:4]
    categorizer = TextCategorizer()

    print(f"Initializing NLP Text Categorization Masterpiece")
    print(f"Mode: {mode}")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")

    if mode == "train_test":
        print("\nCommencing training phase...")
        categorizer.train(train_file)
        print(f"Training complete. Processed {categorizer.total_docs} documents.")
        print(f"Top features selected: {len(categorizer.top_features)}")
        print(f"Categories detected: {', '.join(categorizer.categories)}")

        print("\nInitiating prediction phase...")
        output_file = input("Enter the name for the output file: ")
        categorizer.categorize(test_file, output_file)

        print(f"Prediction complete. Results written to {output_file}")
        log_performance(categorizer.performance.get_stats())

    elif mode == "cross_validate":
        print("\nInitiating cross-validation...")
        accuracies = categorizer.cross_validate(train_file)
        log_performance(categorizer.performance.get_stats(), accuracies)

    else:
        print("Invalid mode. Use 'train_test' or 'cross_validate'.")
        sys.exit(1)

    print("NLP Text Categorization Masterpiece execution complete.")

if __name__ == "__main__":
    main()