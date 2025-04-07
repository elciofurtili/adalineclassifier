import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Função de treinamento do Perceptron
def train_perceptron(X, y, learning_rate=0.1, epochs=100, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples, n_features = X.shape
    # Inicializa os pesos aleatoriamente
    weights = np.random.rand(n_features + 1)  # +1 para o bias
    errors = []

    # Adiciona uma coluna de 1s para o bias
    X_bias = np.c_[np.ones((n_samples, 1)), X]

    for epoch in range(epochs):
        error_count = 0
        for i in range(n_samples):
            # Calcula a saída
            linear_output = np.dot(X_bias[i], weights)
            y_predicted = 1 if linear_output >= 0 else 0
            
            # Atualiza os pesos
            if y[i] != y_predicted:
                weights += learning_rate * (y[i] - y_predicted) * X_bias[i]
                error_count += 1
        
        # Calcula o erro
        errors.append(error_count / n_samples)
    
    return weights, epoch + 1, errors

# Função de teste do Perceptron
def test_perceptron(X, y, weights):
    n_samples = X.shape[0]
    X_bias = np.c_[np.ones((n_samples, 1)), X]
    predictions = np.dot(X_bias, weights) >= 0
    accuracy = np.mean(predictions == y)
    return accuracy

# Função para plotar a fronteira de decisão
def plot_decision_boundary(X, y, weights):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.5)
    
    # Calcula a linha da fronteira de decisão
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.dot(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()], weights)
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Fronteira de Decisão')
    plt.xlabel('Atributo 1')
    plt.ylabel('Atributo 2')
    plt.show()

# Função para executar os experimentos
def run_experiment(train_file, test_file, learning_rate=0.1, epochs=100):
    # Carrega os dados
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    X_train = df_train.drop('label', axis=1).values
    y_train = df_train['label'].values
    X_test = df_test.drop('label', axis=1).values
    y_test = df_test['label'].values
    
    # Treina o Perceptron
    weights, num_epochs, errors = train_perceptron(X_train, y_train, learning_rate, epochs)
    
    # Testa o Perceptron
    train_accuracy = test_perceptron(X_train, y_train, weights)
    test_accuracy = test_perceptron(X_test, y_test, weights)
    
    # Plota a evolução do erro
    plt.plot(errors)
    plt.title('Evolução do Erro de Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Erro')
    plt.show()
    
    # Plota a fronteira de decisão
    plot_decision_boundary(X_train, y_train, weights)
    plot_decision_boundary(X_test, y_test, weights)
    
    print(f'Acurácia no conjunto de treino: {train_accuracy:.2f}')
    print(f'Acurácia no conjunto de teste: {test_accuracy:.2f}')

# Executa os experimentos com os datasets
run_experiment('train_dataset1.csv', 'test_dataset1.csv', learning_rate=0.1, epochs=100)
run_experiment('train_dataset2.csv', 'test_dataset2.csv', learning_rate=0.1, epochs=100)
run_experiment('train_dataset3.csv', 'test_dataset3.csv', learning_rate=0.1, epochs=100)