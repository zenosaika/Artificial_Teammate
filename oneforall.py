from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier

classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Linear SVM': SVC(kernel='linear', C=0.025),
    'Radial Basis Function SVM': SVC(gamma=2, C=1),
    'Guassian Naive Bayes': GaussianNB(),
    'Stochastic Gradient Descent': SGDClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(),
    'Neural Network': MLPClassifier(),
}

digits = datasets.load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

ranking = []

for name, clf in classifiers.items():
    model = make_pipeline(
        StandardScaler(),
        clf
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ranking.append((score, name))

ranking.sort(reverse=True) # sorted by score
for rank, data in enumerate(ranking, 1):
    print(f'[Rank {rank}] {data[0]:.3%} : {data[1]}')
