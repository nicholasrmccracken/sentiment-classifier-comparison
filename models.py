# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        
        for word in sentence:
            word = word.lower()
            if add_to_indexer:
                index = self.indexer.add_and_get_index(word)
            else:
                index = self.indexer.index_of(word)
                
            if index != -1:
                features[index] += 1
                
        return features      

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        
        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            
            if add_to_indexer:
                index = self.indexer.add_and_get_index(bigram)
            else:
                index = self.indexer.index_of(bigram)
                
            if index != -1:
                features[index] += 1
                
        return features   


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor that includes frequency clipping and stop word removal.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "if", "then", "than",
            "is", "am", "are", "was", "were", "be", "been", "being",
            "this", "that", "it", "its", "of", "to", "in", "on", "for",
            "with", "as", "at", "by", "from", "up", "down"
        }
        
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features, counts = Counter(), Counter()
        
        # Remove stopwords
        for word in sentence:
            if word.lower() in self.stop_words: continue
            
            counts[word] += 1
        
        # Frequency clipping
        max_count = 3
        for word, count in counts.items():
            clipped = min(count, max_count)

            if add_to_indexer:
                index = self.indexer.add_and_get_index(word)
            else:
                index = self.indexer.index_of(word)

            if index != -1:
                features[index] += clipped
                
        return features     


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor
    
    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        features = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)
        
        prediction = 0
        for feature_index, feature_count in features.items():
            prediction += self.weights[feature_index] * feature_count
            
        return 1 if prediction >= 0 else 0


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    random.seed(0)
    np.random.seed(0)
    
    # Build vocabulary
    for example in train_exs:
        feat_extractor.extract_features(example.words, add_to_indexer=True)

    # Initialize weights
    vocabulary_size = len(feat_extractor.get_indexer())
    weights = np.zeros(vocabulary_size)
    
    # Hyperparameters
    learning_rate = 0.05
    num_epochs = 20
    
    for _ in range(num_epochs):
        # Randomize order each epoch
        indices = list(range(len(train_exs)))
        random.shuffle(indices)
        
        for index in indices:
            example = train_exs[index]
            features = feat_extractor.extract_features(example.words, add_to_indexer=False)

            # Compute raw score
            prediction = 0
            for feature_index, feature_count in features.items():
                prediction += weights[feature_index] * feature_count
            
            sigmoid_prediction = 1 / (1 + np.exp(-prediction))  # raw score -> probability
            error = example.label - sigmoid_prediction
            
            # Update weights (gradient descent)
            for feature_index, feature_count in features.items():
                weights[feature_index] += learning_rate * error * feature_count
                
    return LogisticRegressionClassifier(weights, feat_extractor)


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class DAN(nn.Module):
    """
    Defines a deep averaging network. This consists of conversion of words to indices, an embedding layer, 
    averaging, and log probability output.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, embedding_layer, hid, out, dropout):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param embedding_layer: embedding layer to convert input tokens to embeddings
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(DAN, self).__init__()
        self.embedding_layer = embedding_layer
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_layer.embedding_dim, hid)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        
        embeddings = self.embedding_layer(x)
        sentence = torch.mean(embeddings, dim=0)
        sentence = self.dropout(sentence)
        
        # Based off FFNN structure
        hidden = self.activation(self.fc1(sentence))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        log_probability = self.log_softmax(logits)
        
        return log_probability
    

class CNN(nn.Module):
    """
    Defines a convultional neural network. This consists of conversion of words to indices, an embedding layer, 
    convolution, global max pool, and log probability output.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, embedding_layer, num_filters, kernel_size, out, dropout):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param embedding_layer: embedding layer to convert input tokens to embeddings
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(CNN, self).__init__()
        self.embedding_layer = embedding_layer
        self.dropout = nn.Dropout(dropout)
        # Padding is necessary in case the word count of the sentence is smaller than the kernel size
        self.conv1 = nn.Conv1d(in_channels=embedding_layer.embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(num_filters, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        
        embeddings = self.embedding_layer(x)
        embeddings = embeddings.transpose(0, 1).unsqueeze(0)  # need to set up to work with conv1d
        
        hidden = self.activation(self.conv1(embeddings))
        pooled = torch.max(hidden, dim=2).values.squeeze(0) # global max pool
        
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        log_probability = self.log_softmax(logits)
        
        return log_probability
    
    
class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        self.network = network
        self.network.eval()
        self.word_embeddings = word_embeddings
    
    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        indexes = []
        for word in ex_words:
            word_index = self.word_embeddings.word_indexer.index_of(word)
            
            if word_index == -1:
                word_index = 1  # UNK
            indexes.append(word_index)
            
        if len(indexes) == 0:
            indexes = [0]  # PAD fallback

        x = torch.tensor(indexes, dtype=torch.long)
        with torch.no_grad():
            log_probs = self.network.forward(x)
            return int(torch.argmax(log_probs).item())


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Hyperparameters 
    num_epochs = args.num_epochs
    hid = args.hidden_size
    lr = args.lr
    dropout = 0.25
    out = 2

    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    dan = DAN(embedding_layer, hid=hid, out=out, dropout=dropout)
    optimizer = optim.Adam(dan.parameters(), lr=lr)

    # For early stopping (very rarely triggered)
    best_dev_accuracy = -1
    best_state = None
    patience = 5
    declining_epochs = 0
    
    for epoch in range(num_epochs):
        # Randomize order each epoch
        indices = list(range(len(train_exs)))
        random.shuffle(indices)
        
        dan.train()  # set to training mode for dropout
        
        total_loss = 0.0
        for index in indices:
            # Convert words to indices
            word_indices = []
            for word in train_exs[index].words:
                word_index = word_embeddings.word_indexer.index_of(word)
                if word_index == -1:  # word is not in vocab
                    word_index = 1
                word_indices.append(word_index)
                
            if len(word_indices) == 0:  # no words, avoid edge case error
                word_indices = [0]

            x = torch.tensor(word_indices, dtype=torch.long)
            y_label = train_exs[index].label
            y_onehot = torch.zeros(out)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y_label, dtype=np.int64)), 1)

            dan.zero_grad()
            log_probability = dan.forward(x)
            loss = torch.neg(log_probability).dot(y_onehot)
            total_loss += loss

            loss.backward()
            optimizer.step()

        # Dev accuracy printing
        dan.eval()  # set to evaluation mode for dropout
        correct = 0
        with torch.no_grad():  # gradients not needed for this part
            for example in dev_exs:
                # Convert words to indices
                word_indices = []
                for word in example.words:
                    word_index = word_embeddings.word_indexer.index_of(word)
                    if word_index == -1:  # word is not in vocab
                        word_index = 1
                    word_indices.append(word_index)
                    
                if len(word_indices) == 0:  # no words, avoid edge case error
                    word_indices = [0]

                x = torch.tensor(word_indices, dtype=torch.long)
                log_probability = dan.forward(x)
                prediction = int(torch.argmax(log_probability).item())
                if prediction == example.label:
                    correct += 1

        dev_accuracy = correct / (len(dev_exs))
        print(f"Epoch {epoch}: total loss = {total_loss:.2f}, dev accuracy = {dev_accuracy:.4f}")
        
        # Check for early stopping
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in dan.state_dict().items()}  # make a clone of state dict for final model
            declining_epochs = 0
        else:
            declining_epochs += 1
            if declining_epochs >= patience:
                break

    # Restore best model weights
    if best_state is not None:
        dan.load_state_dict(best_state)
        
    return NeuralSentimentClassifier(dan, word_embeddings)

def train_convolutional_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your convolutional network model. Nearly identical to deep averaging network training.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Hyperparameters 
    num_epochs = args.num_epochs
    lr = args.lr
    num_filters = 100
    kernel_size = 3
    dropout = 0.25
    out = 2

    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
    cnn = CNN(embedding_layer, num_filters=num_filters, kernel_size=kernel_size, out=out, dropout=dropout)
    optimizer = optim.Adam(cnn.parameters(), lr=lr)

    # For early stopping (very rarely triggered)
    best_dev_accuracy = -1
    best_state = None
    patience = 5
    declining_epochs = 0
    
    for epoch in range(num_epochs):
        # Randomize order each epoch
        indices = list(range(len(train_exs)))
        random.shuffle(indices)
        
        cnn.train()  # set to training mode for dropout
        
        total_loss = 0.0
        for index in indices:
            # Convert words to indices
            word_indices = []
            for word in train_exs[index].words:
                word_index = word_embeddings.word_indexer.index_of(word)
                if word_index == -1:  # word is not in vocab
                    word_index = 1
                word_indices.append(word_index)
                
            if len(word_indices) == 0:  # no words, avoid edge case error
                word_indices = [0]

            x = torch.tensor(word_indices, dtype=torch.long)
            y_label = train_exs[index].label
            y_onehot = torch.zeros(out)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y_label, dtype=np.int64)), 1)

            cnn.zero_grad()
            log_probability = cnn.forward(x)
            loss = torch.neg(log_probability).dot(y_onehot)
            total_loss += loss

            loss.backward()
            optimizer.step()

        # Dev accuracy printing
        cnn.eval()  # set to evaluation mode for dropout
        correct = 0
        with torch.no_grad():  # gradients not needed for this part
            for example in dev_exs:
                # Convert words to indices
                word_indices = []
                for word in example.words:
                    word_index = word_embeddings.word_indexer.index_of(word)
                    if word_index == -1:  # word is not in vocab
                        word_index = 1
                    word_indices.append(word_index)
                    
                if len(word_indices) == 0:  # no words, avoid edge case error
                    word_indices = [0]

                x = torch.tensor(word_indices, dtype=torch.long)
                log_probability = cnn.forward(x)
                prediction = int(torch.argmax(log_probability).item())
                if prediction == example.label:
                    correct += 1

        dev_accuracy = correct / (len(dev_exs))
        print(f"Epoch {epoch}: total loss = {total_loss:.2f}, dev accuracy = {dev_accuracy:.4f}")
        
        # Check for early stopping
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in cnn.state_dict().items()}  # make a clone of state dict for final model
            declining_epochs = 0
        else:
            declining_epochs += 1
            if declining_epochs >= patience:
                break

    # Restore best model weights
    if best_state is not None:
        cnn.load_state_dict(best_state)
        
    return NeuralSentimentClassifier(cnn, word_embeddings)
