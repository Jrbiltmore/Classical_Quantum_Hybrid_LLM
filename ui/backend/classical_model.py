
from sklearn.neural_network import MLPClassifier

class ClassicalNN:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
