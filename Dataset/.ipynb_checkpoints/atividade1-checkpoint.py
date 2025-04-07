import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100, seed=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights = None
        self.errors = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, int, List[float]]:
        np.random.seed(self.seed)
        n_features = X.shape[0]
        n_samples = X.shape[1]
          
        # Adiciona bias no X (valor constante 1)
        X = np.vstack([np.ones((1, n_samples)), X])
        self.weights = np.random.randn(X.shape[0])
          
        for epoch in range(self.epochs):
            total_errors = 0
            for i in range(n_samples):
                prediction = self.predict_single(X[:, i])
                error = y[0, i] - prediction
                if error != 0:
                    total_errors += 1
                    self.weights += self.learning_rate * error * X[:, i]
            self.errors.append(total_errors / n_samples)
            if total_errors == 0:
                break
         
        return self.weights, epoch + 1, self.errors

    def predict_single(self, x: np.ndarray) -> int:
        return 1 if np.dot(self.weights, x) >= 0 else -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[1]
        X = np.vstack([np.ones((1, n_samples)), X])
        predictions = [self.predict_single(X[:, i]) for i in range(n_samples)]
        return np.array(predictions)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(predictions == y.flatten()) * 100


def plot_decision_boundary(perceptron, X, y, title=""):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    X_mesh = np.c_[xx.ravel(), yy.ravel()].T
    Z = perceptron.predict(X_mesh)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), marker='o', edgecolor='k')
    plt.title(title)
    plt.show()


# Função para plotar a evolução do erro
def plot_error_evolution(errors: List[float], title="Error Evolution"):
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.show()

# Carregando Dataset #1
df_train = pd.read_csv('/home/furtili/Documentos/Doutorado/Deep Learning/Atividade 01/Dataset/train_dataset2.csv')
df_test = pd.read_csv('/home/furtili/Documentos/Doutorado/Deep Learning/Atividade 01/Dataset/test_dataset2.csv')

X_train = df_train.drop('label', axis=1).values.T
y_train = df_train['label'].values.reshape(1, -1)
X_test = df_test.drop('label', axis=1).values.T
y_test = df_test['label'].values.reshape(1, -1)

# Treinando o Perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=100)
weights, epochs_run, errors = perceptron.fit(X_train, y_train)

# Plotando a evolução do erro
plot_error_evolution(errors, "Error Evolution - Dataset #1")

# Avaliando o modelo
train_accuracy = perceptron.accuracy(X_train, y_train)
test_accuracy = perceptron.accuracy(X_test, y_test)
print(f'Training Accuracy: {train_accuracy:.2f}%')
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Plotando as fronteiras de decisão para os dados de treino e teste
plot_decision_boundary(perceptron, X_train, y_train, title="Decision Boundary - Training Data")
plot_decision_boundary(perceptron, X_test, y_test, title="Decision Boundary - Test Data")