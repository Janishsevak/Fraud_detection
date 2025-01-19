from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class FraudModel:
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        """
        Trains the RandomForestClassifier model.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained model and logs metrics.
        """
        y_pred = self.model.predict(X_test)
        metrics = {
            "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
            "Accuracy Score": accuracy_score(y_test, y_pred),
            "Classification Report": classification_report(y_test, y_pred, output_dict=True)
        }
        return metrics
