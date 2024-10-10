import os
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

class AdvancedTextCategorizer:
    def __init__(self, use_bigrams=True, feature_threshold=3):
        self.class_priors = defaultdict(float)
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.centroids = defaultdict(lambda: defaultdict(float))
        self.vocab = set()
        self.idf = defaultdict(float)
        self.use_bigrams = use_bigrams
        self.feature_threshold = feature_threshold

    def tokenize(self, text: str) -> List[str]:
        words = text.lower().split()
        tokens = words
        if self.use_bigrams:
            tokens += [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return tokens

    def compute_tfidf(self, documents: List[Tuple[str, str]]):
        df = Counter()
        tf = defaultdict(Counter)
        N = len(documents)

        for doc, _ in documents:
            words = set(self.tokenize(doc))
            df.update(words)
            tf[doc].update(self.tokenize(doc))

        self.idf = {word: math.log(N / (1 + count)) for word, count in df.items()}
        return {doc: {word: count * self.idf[word] for word, count in doc_tf.items()} for doc, doc_tf in tf.items()}

    def train(self, training_file: str):
        print(f"Training on file: {training_file}")
        
        class_counts = Counter()
        documents = []

        with open(training_file, 'r', encoding='utf-8') as f:
            for line in f:
                filepath, category = line.strip().split()
                class_counts[category] += 1
                with open(filepath, 'r', encoding='utf-8') as doc:
                    content = doc.read()
                    documents.append((content, category))

        tfidf_scores = self.compute_tfidf(documents)
        word_counts = defaultdict(Counter)

        for doc, category in documents:
            words = self.tokenize(doc)
            word_counts[category].update(words)
            self.vocab.update(words)

        # Feature selection
        self.vocab = {word for word, count in Counter(word for counts in word_counts.values() for word in counts).items() 
                      if count >= self.feature_threshold}

        # Calculate class priors and word probabilities (Naive Bayes)
        total_docs = sum(class_counts.values())
        for category, count in class_counts.items():
            self.class_priors[category] = math.log(count / total_docs)
            total_words = sum(word_counts[category].values())
            for word in self.vocab:
                count = word_counts[category][word]
                self.word_probs[category][word] = math.log((count + 1) / (total_words + len(self.vocab)))

        # Calculate centroids (Rocchio/TF-IDF)
        for category in class_counts:
            category_docs = [doc for doc, cat in documents if cat == category]
            self.centroids[category] = {word: sum(tfidf_scores[doc].get(word, 0) for doc in category_docs) / len(category_docs)
                                        for word in self.vocab}

        print(f"Training completed. Processed {len(documents)} documents with {len(self.vocab)} features.")

    def predict(self, document: str) -> str:
        words = self.tokenize(document)
        
        # Naive Bayes prediction
        nb_scores = {category: prior for category, prior in self.class_priors.items()}
        for category in nb_scores:
            nb_scores[category] += sum(self.word_probs[category].get(word, 0) for word in words if word in self.vocab)

        # Rocchio/TF-IDF prediction
        doc_tfidf = Counter(words)
        centroid_scores = {category: sum(self.centroids[category].get(word, 0) * doc_tfidf[word] * self.idf.get(word, 0)
                                         for word in set(words) & self.vocab)
                           for category in self.centroids}

        # Combine scores
        combined_scores = {category: nb_scores[category] + centroid_scores[category] 
                           for category in nb_scores}

        return max(combined_scores, key=combined_scores.get)

    def evaluate(self, test_file: str, output_file: str = None):
        print(f"Evaluating on file: {test_file}")
        
        predictions = []
        true_labels = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                filepath = parts[0]
                true_category = parts[1] if len(parts) > 1 else None
                try:
                    with open(filepath, 'r', encoding='utf-8') as doc:
                        content = doc.read()
                        prediction = self.predict(content)
                        predictions.append((filepath, prediction))
                        if true_category:
                            true_labels.append(true_category)
                except FileNotFoundError:
                    print(f"Warning: File not found: {filepath}")
                    continue

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for filepath, prediction in predictions:
                    f.write(f"{filepath} {prediction}\n")
            print(f"Predictions written to {output_file}")
        
        if true_labels:
            correct = sum(1 for (_, pred), true in zip(predictions, true_labels) if pred == true)
            accuracy = correct / len(true_labels)
            print(f"Accuracy: {accuracy:.4f}")

        print(f"Evaluation completed. Processed {len(predictions)} documents.")

def main():
    classifier = AdvancedTextCategorizer()

    train_file = input("Enter the name of the training file: ")
    classifier.train(train_file)
    
    test_file = input("Enter the name of the test file: ")
    
    if "corpus1" in train_file:
        output_file = input("Enter the name for the output file: ")
        classifier.evaluate(test_file, output_file)
        print("\nText categorization complete.")
        print(f"Predictions written to {output_file}")
    else:
        classifier.evaluate(test_file)
        print("\nText categorization complete.")

if __name__ == "__main__":
    main()