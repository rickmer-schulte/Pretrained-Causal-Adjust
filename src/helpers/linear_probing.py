# helpers/linear_probing.py

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


def train_linear_probe(X, y, type="logistic", test_size=0.2, random_state=42, max_iter=1000):
    """
    Train a logistic regression model to perform linear probing on the features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N, D).
    y : np.ndarray
        Labels of shape (N,).
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed.
    max_iter : int
        Maximum number of iterations for the solver.

    Returns
    -------
    clf : sklearn.linear_model.LogisticRegression
        Trained logistic regression model.
    acc : float
        Accuracy of the model on the test set.
    X_train, X_test, y_train, y_test : np.ndarray
        Split datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    if type == "logistic":
        model = LogisticRegression(max_iter=max_iter, random_state=random_state, penalty=None) 
    if type == "logistic-l2":
        model = LogisticRegression(max_iter=max_iter, random_state=random_state)   
    elif type == "linear":
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if type == "logistic":
        acc = accuracy_score(y_test, y_pred)
    elif type == "linear":
        acc = mean_squared_error(y_test, y_pred)
    
    return model, acc, X_train, X_test, y_train, y_test