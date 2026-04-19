from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from sklearn.utils.validation import check_X_y, check_array


class ModelTrainer:
    """A class for training and evaluating machine learning models."""

    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters

    def train(self, X_train, y_train):
        X_train, y_train = check_X_y(X_train, y_train, ensure_all_finite=True)
        self.model.set_params(**self.parameters)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        X_test = check_array(X_test, ensure_all_finite=True)
        y_test = check_array(y_test, ensure_2d=False, ensure_all_finite=True)

        predictions = self.model.predict(X_test)

        prob_predictions = (
            self.model.predict_proba(X_test)[:, 1]
            if hasattr(self.model, "predict_proba")
            else [0] * len(y_test)
        )

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, prob_predictions)
        recall = recall_score(y_test, predictions)

        return accuracy, f1, roc_auc, recall, predictions