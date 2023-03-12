from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

digits = datasets.load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = MLPClassifier(batch_size=256, verbose=True, early_stopping=True)

model = make_pipeline(
    StandardScaler(),
    clf
)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

parameters = {
    'solver': ['sgd', 'adam', 'lbfgs'],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': uniform(0.0001, 0.9),
    'learning_rate': ['constant','adaptive'],
}

random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions = parameters,
        cv = 2, 
        n_iter = 10, 
        n_jobs=-1                        
    )
random_search.fit(X_train, y_train)

print(f'Optimized parameters {random_search.best_params_}')
print(f'Baseline score : {score:.3%}')
print(f'Optimized score {random_search.best_score_:.3%}')
