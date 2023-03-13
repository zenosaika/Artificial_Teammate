import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

digits = datasets.load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

########################################################################

random_forest_hyperparams = {
    # number of trees in the random forest
    'n_estimators': [int(x) for x in np.linspace(start = 1, stop = 20, num = 20)],
    # number of features in consideration at every split
    'max_features': ['auto', 'sqrt'],
    # maximum number of levels allowed in each decision tree
    'max_depth': [int(x) for x in np.linspace(10, 120, num = 12)],
    # minimum sample number to split a node
    'min_samples_split': [2, 6, 10],
    # minimum sample number that can be stored in a leaf node
    'min_samples_leaf': [1, 3, 4],
    # method used to sample data points
    'bootstrap': [True, False],
}

neural_network_hyperparams = {
    'solver': ['sgd', 'adam', 'lbfgs'],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': uniform(0.0001, 0.9),
    'learning_rate': ['constant','adaptive'],
}

########################################################################

classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Neural Network': MLPClassifier(),
}

parameters = {
    'Random Forest': random_forest_hyperparams,
    'Neural Network': neural_network_hyperparams,
}

results = []

for name, clf in classifiers.items():
    model = make_pipeline(
        StandardScaler(),
        clf
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    # Hyperparameters Tuning
    random_search = RandomizedSearchCV(
        estimator = clf,
        param_distributions = parameters[name],
        cv = 5, 
        verbose = 2,
        n_iter = 100, 
        n_jobs = -1,                      
    )
    random_search.fit(X_train, y_train)

    # print(f'[{name}] Optimized parameters {random_search.best_params_}')
    # print(f'[{name}] Baseline score : {score:.3%}')
    # print(f'[{name}] Optimized score {random_search.best_score_:.3%}')
    results.append((f'[{name}] Baseline score : {score:.3%}', 
                    f'[{name}] Optimized score {random_search.best_score_:.3%}'
                    ))

for result in results:
    print(result[0])
    print(result[1])
    