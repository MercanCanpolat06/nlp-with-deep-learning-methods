### Linear Regression

import numpy as np
import json

def load_data(vector_file, labels_file):
    with open(labels_file, "r") as f:
        labels_dict = json.load(f)
    
    X, y = [], []
    
    print("Matching vectors and labels")
    with open(vector_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            
            if word in labels_dict:
                vector = np.array(parts[1:], dtype=np.float32)
                X.append(vector)
                y.append(labels_dict[word])
                
    return np.array(X), np.array(y).reshape(-1, 1)

X, y = load_data("../data/my_custom_vectors.txt", "qwen_labels.json")
m, n = X.shape  # m: example number, n: number of parameters
print(f"Data uploaded {m} examples, {n} dimension.")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_test_split(X, y, test_size=0.2):
    m = X.shape[0]
    
    indices = np.arange(m)
    np.random.seed(42) 
    np.random.shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    test_count = int(m * test_size)
    train_count = m - test_count
    X_train, X_test = X_shuffled[:train_count], X_shuffled[train_count:]
    y_train, y_test = y_shuffled[:train_count], y_shuffled[train_count:]
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
m_train, n = X_train.shape

#choose weight and bias zeros
W = np.zeros((n, 1))
b = 0

verb_weight = 3

# Hyperparameters
learning_rate = 0.1
epochs = 10000

print("Training is started")

for epoch in range(epochs):
    # Train
    z_train = np.dot(X_train, W) + b
    y_hat_train = sigmoid(z_train)
    loss = -np.mean(verb_weight * y_train * np.log(y_hat_train + 1e-15) + (1 - y_train) * np.log(1 - y_hat_train + 1e-15))
    
    error = y_hat_train - y_train
    weighted_error = error * (1 + (verb_weight - 1) * y_train)
    dw = (1 / m_train) * np.dot(X_train.T, weighted_error)
    db = (1 / m_train) * np.sum(weighted_error)
    
    W -= learning_rate * dw
    b -= learning_rate * db
    
    # TEST 
    if epoch % 500 == 0:
        z_test = np.dot(X_test, W) + b
        y_hat_test = sigmoid(z_test)
        loss_test = -np.mean(y_test * np.log(y_hat_test + 1e-15) + (1 - y_test) * np.log(1 - y_hat_test + 1e-15))
        
        
        predictions_test = (y_hat_test >= 0.5).astype(int)
        accuracy_test = np.mean(predictions_test.flatten() == y_test.flatten())
        
        print(f"Epoch {epoch}: Train Loss = {loss:.4f} | Test Loss = {loss_test:.4f} | Test Acc = {accuracy_test:.2%}")

print("Training completed")


from sklearn.metrics import confusion_matrix, classification_report

# Test seti tahminlerini al
z_final = np.dot(X_test, W) + b
y_pred_final = (sigmoid(z_final) >= 0.5).astype(int)

print("--- Confusion matrix ---")
print(confusion_matrix(y_test, y_pred_final))
print("\n--- Report ---")
print(classification_report(y_test, y_pred_final))