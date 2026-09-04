"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
def clean_text(text: str) -> str:
    # TODO: Lowercase text and replace non-alphabetic chars with spaces
    list_str = [c.lower() if c.isalpha() else ' ' for c in text]
    return ("".join(list_str)).strip()

# Step 2 - tokenize
def tokenize(text: str) -> list:
    # TODO: Split cleaned text on whitespace into non-empty word tokens
    return text.split()

# Step 3 - tokenize_corpus
def clean_text(text: str) -> str:
    # TODO: Lowercase text and replace non-alphabetic chars with spaces
    list_str = [c.lower() if c.isalpha() else ' ' for c in text]
    return ("".join(list_str)).strip()
    
def tokenize_corpus(texts: list) -> list:
    # TODO: Apply clean_text and tokenize to every document so the full corpus becomes a list of token lists.
    return [clean_text(elem).split() for elem in texts]

# Step 4 - split_train_val_test_indices
def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    # TODO: Produce shuffled index arrays that partition n_samples into train/val/test
    np.random.seed(seed)
    idxs = np.arange(0,n_samples)
    np.random.shuffle(idxs)
    n_val = int(n_samples*val_fraction)
    n_test = int(n_samples*test_fraction)
    test_idxs = idxs[:n_test]
    val_idxs = idxs[n_test:n_test+n_val]
    train_idxs = idxs[n_test+n_val:]
    return train_idxs, val_idxs, test_idxs,

# Step 5 - count_word_frequencies
from collections import Counter
import numpy as np

def count_word_frequencies(tokenized_docs: list) -> dict:
    toc_docs_array = [elem for doc in tokenized_docs for elem in doc]
    return dict(Counter(toc_docs_array))

# Step 6 - build_vocabulary
from collections import Counter
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    # TODO: Keep the top max_size most frequent words; map each to an index in [0, V).
    count_dict = Counter(word_counts)
    k = min(max_size, len(count_dict))
    list_of_common_pair = count_dict.most_common(k) # (the, 5)
    list_of_common_pair = sorted(list_of_common_pair, key=lambda x:(-x[1], x[0]))
    vocab = {elem[0]:ind for ind, elem in enumerate(list_of_common_pair)}


    return vocab

# Step 7 - tokens_to_bow
from collections import Counter
def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    # TODO: Convert one document's token list into a bag-of-words count vector...
    res = np.zeros(len(vocab))
    counter_dct = Counter(tokens)
    #for word, index in vocab.items():
    #    if word in counter_dct:
    #        res[index] = counter_dct[word]

    real_words = [word for word in counter_dct if word in vocab]
    indices = [vocab[word] for word in real_words] 
    counts = [counter_dct[word] for word in real_words]
    res[indices] = counts

    return res

# Step 8 - corpus_to_bow_matrix
def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    # TODO: Stack per-document BoW vectors into a 2-D count matrix for a whole corpus.
    if len(tokenized_docs) == 0:
        return np.empty((0, len(vocab)))

    res = [tokens_to_bow(doc, vocab).tolist() for doc in tokenized_docs]
    return np.asarray(res)

# Step 9 - compute_document_frequencies
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    # TODO: Count docs where each term appears at least once (df, shape (V,))
    return np.sum((bow_matrix != 0),axis = 0)

# Step 10 - compute_idf
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # TODO: Compute smoothed IDF idf_j = log((n_docs + 1) / (df_j + 1)) + 1
    res = (n_docs + 1)/(df + 1)
    return np.log(res) + 1

# Step 11 - transform_tfidf
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # TODO: Multiply BoW counts by the fitted IDF vector to produce TF-IDF features.
    return bow_matrix*idf

# Step 12 - fit_tfidf
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    # TODO: Fit IDF on the training BoW matrix by chaining DF and IDF.
    df = compute_document_frequencies(bow_train)
    idf = compute_idf(df, bow_train.shape[0])
    return idf

# Step 13 - sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    # TODO: Map logits to probabilities with a numerically stable logistic sigmoid.
    
    answ = np.where(z>=0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))    
    return answ

# Step 14 - logistic_predict_proba
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # TODO: Return P(y=1|x) for each row via linear scores and sigmoid
    answers = sigmoid(X@w + b)
    return answers

# Step 15 - binary_cross_entropy
def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    # TODO: Compute mean binary cross-entropy plus L2 penalty on the weights.
    loss = -np.mean(y_true*np.log(y_prob)) -np.mean((1-y_true)*np.log(1-y_prob))
    reg = l2_lambda*np.sum(w**2)/2
    return loss + reg

# Step 16 - logistic_gradients
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    Args:
        X: Feature matrix of shape (N, D).
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        w: Weight vector of shape (D,).
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple (dw, db) with dw shape (D,) and db a float.
    """
    dw = ((y_prob - y_true)@X)/X.shape[0] + l2_lambda*w
    db = float(np.mean((y_prob - y_true)))
    return (dw, db)

# Step 17 - initialize_logistic_params
def initialize_logistic_params(n_features: int) -> tuple:
    # TODO: Return a zero weight vector of shape (n_features,) and bias 0.0
    return (np.zeros(n_features,), 0.0)

# Step 18 - gradient_descent_step
def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:
    # TODO: Run one full-batch gradient descent update; return (w_new, b_new, loss).
    y_pred = sigmoid(X@w + b)
    dw, db = logistic_gradients(X, y, y_pred, w, l2_lambda)
    w_new = w - lr*dw
    b_new = b - lr*db
    loss = binary_cross_entropy(y, y_pred, w, l2_lambda)
    return (w_new, b_new, loss)

# Step 19 - train_logistic_regression
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    # TODO: Initialize params and run n_epochs of full-batch GD, recording loss...
    w, b = initialize_logistic_params(X.shape[1])
    losses = []
    for epoch_n in range(n_epochs):
        w, b, loss = gradient_descent_step(X, y, w, b, lr, l2_lambda)
        losses.append(loss)

    return w, b, losses

# Step 20 - predict_labels
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    res = np.where(proba>=threshold, 1, 0)
    return res

# Step 21 - confusion_counts
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    # TODO: Return the four confusion-matrix counts (tp, fp, tn, fn) as Python ints
    tp = np.sum((y_true == 1) & (y_pred == 1)).item()
    fp = np.sum((y_true == 0) & (y_pred == 1)).item()
    tn = np.sum((y_true == 0) & (y_pred == 0)).item()
    fn = np.sum((y_true == 1) & (y_pred == 0)).item()
    return tp, fp, tn, fn

# Step 22 - metrics_from_counts
def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    # TODO: Derive precision, recall, F1, and accuracy from confusion counts...
    
    res_dict = {}
    eps = 1e-9
    res_dict['precision'] = tp / (tp + fp +eps)
    res_dict['recall'] = tp / (tp + fn +eps)
    res_dict['f1'] = 2*res_dict['precision']*res_dict['recall']/(res_dict['precision'] + res_dict['recall'] + eps)
    res_dict['accuracy'] = (tp + tn)/(tp + fp + tn +fn +eps)


    return res_dict

# Step 23 - tune_decision_threshold
def tune_decision_threshold(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray = None) -> tuple:
    # TODO: Find the decision threshold that maximizes F1 on validation data.
    best_f1 = 0
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    best_treshold = thresholds[0]
    for threshold in thresholds:
        labels = predict_labels(proba, threshold)
        confusions = confusion_counts(y_true, labels)
        f1 = metrics_from_counts(*confusions)['f1']
        if (f1 > best_f1):
            best_f1 = f1
            best_treshold = threshold
    
    return best_treshold, best_f1

# Step 24 - evaluate_predictions
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # TODO: Bundle confusion counts and classification metrics into one report dict
    confusions = confusion_counts(y_true, y_pred)

    metric_dict = metrics_from_counts(*confusions)
    conf_tuples = zip( ['tp', 'fp', 'tn', 'fn'], confusions)
    conf_dict = dict(conf_tuples)
    return {**metric_dict, **conf_dict}

# Step 25 - vectorize_texts
def vectorize_texts(texts: list, vocab: dict, idf: np.ndarray) -> np.ndarray:
    # TODO: Clean, tokenize, BoW, and TF-IDF transform a list of raw strings.
    tokenized_texts = tokenize_corpus(texts)
    bow_matrix = corpus_to_bow_matrix(tokenized_texts, vocab) # сколько раз слово из словаря встретилось в документе
    return transform_tfidf(bow_matrix,idf)

# Step 26 - predict_text
def predict_text(text: str, vocab: dict, idf: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> int:
    """Label a single raw message with the fitted classifier.

    Args:
        text: Raw input string.
        vocab: Fitted word -> column index map.
        idf: Fitted IDF vector, shape (V,).
        w: Logistic weight vector, shape (V,).
        b: Logistic bias scalar.
        threshold: Decision threshold for the positive class.

    Returns:
        Predicted label as int 0 or 1.
    """
    # TODO: label a single unseen raw message using fitted model artifacts
    X = vectorize_texts([text], vocab, idf) # (1, V)
    prob = sigmoid(X@w + b)
    return (predict_labels(prob, threshold).item())

# Step 27 - collect_prediction_errors
def collect_prediction_errors(texts: list, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # TODO: Gather raw messages that are false positives vs false negatives...
    false_positives_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    false_positives_texts = [text for i, text in enumerate(texts) if i in false_positives_indices]
    false_negatives_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    false_negatives_texts = [text for i, text in enumerate(texts) if i in false_negatives_indices]
    return {'false_positives': false_positives_texts, 'false_negatives': false_negatives_texts}

