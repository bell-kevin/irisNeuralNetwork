import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------
# 1. Load and prepare the Iris dataset
# ------------------------------------------------------------
iris = datasets.load_iris()
X = iris.data  # shape: (150, 4)
y = iris.target.reshape(-1, 1)  # shape: (150, 1)

# Convert labels to one-hot vectors, e.g., 0 -> [1, 0, 0], 1 -> [0, 1, 0], etc.
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)  # shape: (150, 3)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, 
                                                    test_size=0.2, 
                                                    random_state=42)

# ------------------------------------------------------------
# 2. Define hyperparameters
# ------------------------------------------------------------
input_size = 4    # Iris has 4 features
hidden_size1 = 8
hidden_size2 = 8
output_size = 3   # 3 classes (setosa, versicolor, virginica)
learning_rate = 0.01
epochs = 1000

# ------------------------------------------------------------
# 3. Initialize network parameters
# ------------------------------------------------------------
# Weights shape:
#   W1: (4, 8),   b1: (1, 8)
#   W2: (8, 8),   b2: (1, 8)
#   W3: (8, 3),   b3: (1, 3)
np.random.seed(42)  # for reproducibility

W1 = np.random.randn(input_size, hidden_size1) * 0.01
b1 = np.zeros((1, hidden_size1))

W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
b2 = np.zeros((1, hidden_size2))

W3 = np.random.randn(hidden_size2, output_size) * 0.01
b3 = np.zeros((1, output_size))

# ------------------------------------------------------------
# 4. Activation functions
# ------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    # derivative of ReLU: 1 for z>0 else 0
    return (z > 0).astype(float)

def softmax(z):
    # numerically stable softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ------------------------------------------------------------
# 5. Loss function (cross-entropy) and its derivative
# ------------------------------------------------------------
def cross_entropy_loss(pred, true):
    # pred and true both shape: (N, 3), where N is batch size
    # Add a small epsilon to avoid log(0).
    eps = 1e-15
    return -np.mean(np.sum(true * np.log(pred + eps), axis=1))

def cross_entropy_deriv(pred, true):
    # derivative of cross-entropy w.r.t. softmax output
    # pred and true shape: (N, 3)
    return (pred - true) / pred.shape[0]

# ------------------------------------------------------------
# 6. Training loop
# ------------------------------------------------------------
for epoch in range(epochs):
    # ---------- Forward pass ----------
    # Layer 1
    z1 = X_train.dot(W1) + b1          # shape: (N, 8)
    a1 = relu(z1)                      # shape: (N, 8)

    # Layer 2
    z2 = a1.dot(W2) + b2               # shape: (N, 8)
    a2 = relu(z2)                      # shape: (N, 8)

    # Output layer
    z3 = a2.dot(W3) + b3               # shape: (N, 3)
    a3 = softmax(z3)                   # shape: (N, 3), final predictions

    # ---------- Compute loss ----------
    loss = cross_entropy_loss(a3, y_train)
    
    # ---------- Backward pass ----------
    # dL/dz3
    d_z3 = cross_entropy_deriv(a3, y_train)  # shape: (N, 3)

    # dL/dW3 and dL/db3
    dW3 = a2.T.dot(d_z3)  # shape: (8, 3)
    db3 = np.sum(d_z3, axis=0, keepdims=True)  # shape: (1, 3)

    # dL/da2
    d_a2 = d_z3.dot(W3.T)  # shape: (N, 8)

    # dL/dz2
    d_z2 = d_a2 * relu_deriv(z2)

    # dL/dW2 and dL/db2
    dW2 = a1.T.dot(d_z2)  # shape: (8, 8)
    db2 = np.sum(d_z2, axis=0, keepdims=True)  # shape: (1, 8)

    # dL/da1
    d_a1 = d_z2.dot(W2.T)  # shape: (N, 8)

    # dL/dz1
    d_z1 = d_a1 * relu_deriv(z1)

    # dL/dW1 and dL/db1
    dW1 = X_train.T.dot(d_z1)  # shape: (4, 8)
    db1 = np.sum(d_z1, axis=0, keepdims=True)  # shape: (1, 8)

    # ---------- Update weights ----------
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # (Optional) print loss every 100 iterations
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# ------------------------------------------------------------
# 7. Evaluate on the test set
# ------------------------------------------------------------
def predict(X_data):
    # forward pass only
    z1_ = X_data.dot(W1) + b1
    a1_ = relu(z1_)

    z2_ = a1_.dot(W2) + b2
    a2_ = relu(z2_)

    z3_ = a2_.dot(W3) + b3
    a3_ = softmax(z3_)

    # return class index with highest probability
    return np.argmax(a3_, axis=1)

y_test_indices = np.argmax(y_test, axis=1)
y_pred_indices = predict(X_test)

accuracy = np.mean(y_test_indices == y_pred_indices)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")
