import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import math
import random

nltk.download('punkt_tab')

print("NLTK Data Path:", nltk.data.path)
print("Punkt available:", os.path.exists(os.path.join(nltk.data.find("tokenizers/punkt"), "english.pickle")))

class TextCategorizer:
    def __init__(self):
        self.categories = set()
        self.vocab = set()
        self.cat_doc_counts = defaultdict(int)
        self.cat_word_counts = defaultdict(lambda: defaultdict(int))
        self.total_docs = 0
        
        # Download necessary NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        sentences = sent_tokenize(text.lower())
        tokens = []
        for sentence in sentences:
            tokens.extend(word_tokenize(sentence))
        tokens = [self.stemmer.stem(token) for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def train(self, train_data):
        self.categories.clear()
        self.vocab.clear()
        self.cat_doc_counts.clear()
        self.cat_word_counts.clear()
        self.total_docs = 0

        for filepath, category in train_data:
            self.categories.add(category)
            self.cat_doc_counts[category] += 1
            self.total_docs += 1
            
            with open(filepath, 'r', encoding='utf-8') as doc_file:
                text = doc_file.read()
                tokens = self.preprocess(text)
                self.vocab.update(tokens)
                for token in tokens:
                    self.cat_word_counts[category][token] += 1

    def predict(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = self.preprocess(text)
        
        scores = {}
        for category in self.categories:
            score = math.log(self.cat_doc_counts[category] / self.total_docs)
            for token in tokens:
                if token in self.vocab:
                    count = self.cat_word_counts[category][token]
                    score += math.log((count + 1) / (sum(self.cat_word_counts[category].values()) + len(self.vocab)))
            scores[category] = score
        
        return max(scores, key=scores.get)

    def categorize(self, test_data):
        predictions = []
        for filepath, _ in test_data:
            category = self.predict(filepath)
            predictions.append((filepath, category))
        return predictions

    def cross_validate(self, all_data, k=5):
        random.shuffle(all_data)
        fold_size = len(all_data) // k
        accuracies = []

        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(all_data)
            test_data = all_data[start:end]
            train_data = all_data[:start] + all_data[end:]

            self.train(train_data)
            predictions = self.categorize(test_data)
            
            correct = sum(1 for (_, pred), (_, true) in zip(predictions, test_data) if pred == true)
            accuracy = correct / len(test_data)
            accuracies.append(accuracy)

        return accuracies

def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            filepath, category = line.strip().rsplit(' ', 1)
            data.append((filepath, category))
    return data

def main():
    print("NLTK Data Path:", nltk.data.path)
    print("Punkt available:", nltk.data.find('tokenizers/punkt'))
    
    categorizer = TextCategorizer()
    
    mode = input("Enter mode (train_test / cross_validate): ").strip().lower()
    
    if mode == "train_test":
        train_file = input("Enter the path to the training file: ")
        test_file = input("Enter the path to the test file: ")
        output_file = input("Enter the path for the output file: ")

        train_data = load_data(train_file)
        categorizer.train(train_data)

        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append((line.strip(), None))

        predictions = categorizer.categorize(test_data)

        with open(output_file, 'w', encoding='utf-8') as f_out:
            for filepath, category in predictions:
                f_out.write(f"{filepath} {category}\n")

        print(f"Categorization complete. Results written to {output_file}")

    elif mode == "cross_validate":
        data_file = input("Enter the path to the data file: ")
        all_data = load_data(data_file)
        
        accuracies = categorizer.cross_validate(all_data)
        print(f"Cross-validation accuracies: {accuracies}")
        print(f"Average accuracy: {sum(accuracies) / len(accuracies):.4f}")

    else:
        print("Invalid mode selected. Please choose 'train_test' or 'cross_validate'.")

if __name__ == "__main__":
    main()