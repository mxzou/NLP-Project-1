import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import math
import os
import random
import logging
from typing import List, Tuple, Dict

# Set up logging for better debugging and progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NaiveBayesClassifier:
    def __init__(self):
        # Initialize data structures for the classifier
        self.vocab = set()  # Set of all unique words in the training data
        self.class_word_counts = defaultdict(lambda: defaultdict(int))  # Count of each word in each class
        self.class_totals = defaultdict(int)  # Total word count for each class
        self.class_doc_counts = defaultdict(int)  # Number of documents in each class
        self.total_docs = 0  # Total number of documents
        self.stemmer = PorterStemmer()  # For word stemming
        self.stop_words = set(stopwords.words('english'))  # Common words to ignore

    def preprocess(self, text: str) -> List[str]:
        """Tokenize, lowercase, remove stopwords, and stem the input text."""
        tokens = word_tokenize(text.lower())
        return [self.stemmer.stem(token) for token in tokens if token.isalnum() and token not in self.stop_words]

    def train(self, documents: List[Tuple[str, str]]):
        """Train the classifier on the given documents."""
        logging.info("Starting training process...")
        self.total_docs = len(documents)
        
        # Process each document
        for filepath, category in documents:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as doc_file:
                text = doc_file.read()
            tokens = self.preprocess(text)
            self.vocab.update(tokens)
            for token in tokens:
                self.class_word_counts[category][token] += 1
                self.class_totals[category] += 1
            self.class_doc_counts[category] += 1
        
        # Precompute log probabilities for efficiency during classification
        self.log_priors = {cat: math.log(count / self.total_docs) for cat, count in self.class_doc_counts.items()}
        self.log_likelihoods = defaultdict(lambda: defaultdict(float))
        for category, word_counts in self.class_word_counts.items():
            total_words = self.class_totals[category]
            vocab_size = len(self.vocab)
            for word, count in word_counts.items():
                # Use Laplace smoothing (add-one) to handle unseen words
                self.log_likelihoods[category][word] = math.log((count + 1) / (total_words + vocab_size))
        
        logging.info(f"Training complete. Vocabulary size: {len(self.vocab)}")

    def classify(self, text: str) -> str:
        """Classify the given text using the trained model."""
        tokens = self.preprocess(text)
        scores = {category: self.log_priors[category] for category in self.class_doc_counts}
        
        # Calculate the score for each category
        for category in scores:
            for token in tokens:
                if token in self.vocab:
                    # Use precomputed log likelihoods, or calculate for unseen words
                    scores[category] += self.log_likelihoods[category].get(token, math.log(1 / (self.class_totals[category] + len(self.vocab))))
        
        # Return the category with the highest score
        return max(scores, key=scores.get)

def load_corpus(filename: str, is_training: bool = True) -> List[Tuple[str, str]]:
    """
    Load the corpus from the given file.
    :param filename: Path to the corpus file
    :param is_training: Whether this is a training file (with categories) or test file
    :return: List of (filepath, category) tuples
    """
    documents = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().rsplit(' ', 1)
            if is_training:
                filepath, category = parts
            else:
                filepath = parts[0]
                category = ''  # Empty string for test documents
            documents.append((filepath, category))
    return documents

def k_fold_cross_validation(documents: List[Tuple[str, str]], k: int = 5) -> float:
    """Perform k-fold cross-validation on the given documents."""
    random.shuffle(documents)
    fold_size = len(documents) // k
    total_accuracy = 0
    
    for i in range(k):
        logging.info(f"Starting fold {i+1}/{k}")
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else len(documents)
        
        # Split data into training and test sets
        test_set = documents[test_start:test_end]
        train_set = documents[:test_start] + documents[test_end:]
        
        # Train and evaluate the classifier
        classifier = NaiveBayesClassifier()
        classifier.train(train_set)
        
        correct = sum(1 for filepath, true_category in test_set 
                      if classifier.classify(open(filepath, 'r', encoding='utf-8', errors='ignore').read()) == true_category)
        
        accuracy = correct / len(test_set)
        total_accuracy += accuracy
        logging.info(f"Fold {i+1} accuracy: {accuracy:.4f}")
    
    avg_accuracy = total_accuracy / k
    logging.info(f"Average accuracy across {k} folds: {avg_accuracy:.4f}")
    return avg_accuracy

def main():
    # Prompt user for input files
    train_file = input("Enter the path to the training file: ")
    test_file = input("Enter the path to the test file: ")
    
    # Load training documents
    train_documents = load_corpus(train_file, is_training=True)
    logging.info(f"Loaded {len(train_documents)} training documents from {train_file}")
    
    # Train the classifier
    classifier = NaiveBayesClassifier()
    classifier.train(train_documents)
    
    # Process test documents
    test_documents = load_corpus(test_file, is_training=False)
    
    logging.info(f"Classifying {len(test_documents)} test documents")
    predictions = []
    for filepath, _ in test_documents:
        full_path = os.path.join(os.path.dirname(test_file), filepath)
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as doc_file:
            text = doc_file.read()
        predicted_category = classifier.classify(text)
        predictions.append((filepath, predicted_category))
    
    # Prompt user for output file
    output_file = input("Enter the name of the output file: ")
    
    # Write predictions to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for filepath, category in predictions:
            f.write(f"{filepath} {category}\n")
    
    logging.info(f"Predictions written to {output_file}")
    
    # If this is corpus2 or corpus3, perform cross-validation
    if "corpus2" in train_file or "corpus3" in train_file:
        logging.info("Performing k-fold cross-validation")
        avg_accuracy = k_fold_cross_validation(train_documents)
        print(f"Cross-validation average accuracy: {avg_accuracy:.4f}")

if __name__ == "__main__":
    main()
