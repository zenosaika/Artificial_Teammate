import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

digits = datasets.load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print(f'score: {clf.score(X_test, y_test):.3%}')

fig = plt.figure(figsize=(50, 25))
plot_tree(
    clf, 
    feature_names = [str(i) for i in digits.feature_names], 
    class_names = [str(i) for i in digits.target_names],
    filled = True,
    fontsize = 10,
    )
fig.savefig('decision_tree.png')
