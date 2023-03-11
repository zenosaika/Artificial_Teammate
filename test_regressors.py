from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Regressor
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

classifiers = {
    'Linear Regression': LinearRegression(),
    'XGBoost': XGBRegressor(),
    'Stochastic Gradient Descent': SGDRegressor(),
    'KernelRidge': KernelRidge(),
    'ElasticNet': ElasticNet(),
    'BayesianRidge': BayesianRidge(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Linear SVM': LinearSVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_split=2
    ),
    'Neural Network': MLPRegressor(
        solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(10, 2), max_iter=1000
    )
}

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

ranking = []

for name, clf in classifiers.items():
    model = make_pipeline(clf)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ranking.append((score, name))

ranking.sort(reverse=True) # sorted by score
for rank, data in enumerate(ranking, 1):
    print(f'[Rank {rank}] {data[0]:.3%} : {data[1]}')
