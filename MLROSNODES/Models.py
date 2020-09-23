from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from Processing import print_confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def train_DTC(x, y):
    decision_tree_classifier = tree.DecisionTreeClassifier()
    decision_tree_classifier.fit(x, y)
    return decision_tree_classifier


def train_berNB(x, y):
    bernoulli_nb = BernoulliNB()
    bernoulli_nb.fit(x, y)
    return bernoulli_nb


def train_GNB(x, y):
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(x, y)
    return gaussian_nb


def train_LSVC(x, y):
    linear_svc = LinearSVC()
    linear_svc.fit(x, y)
    return linear_svc


def train_RFC(x, y):
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(x, y)
    return random_forest_classifier


def train_LRC(x, y):
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x, y)
    return logistic_regression


def train_KNC(x, y):
    k_neighbors_classifier = KNeighborsClassifier(n_neighbors=1)
    k_neighbors_classifier.fit(x, y)
    return k_neighbors_classifier


def train_LDA(x, y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x, y)
    return lda


def train_GBC(x, y):
    gbc = GradientBoostingClassifier()
    gbc.fit(x, y)
    return gbc


def MMC():
    return None


def accuracy_score(model, train_x, train_y):
    k_fold_scores = cross_val_score(model, train_x, train_y, cv=10, scoring='accuracy')
    model_score = model.score(train_x, train_y)
    print("Model Accuracy Score : %f" % model_score)
    print("Model Accuracy Average Score of K Fold : %f" % k_fold_scores.mean())
    return model_score


def initialize_confusion_graph(name, model, test, label_encoder):
    prediction = model.predict(test.iloc[:, :-1].values)
    con_matrix = confusion_matrix(
        label_encoder.inverse_transform(prediction),
        label_encoder.inverse_transform(test.iloc[:, -1].values),
        ['valid', 'invalid']
    )
    print_confusion_matrix(name, con_matrix, ['valid', 'invalid'])
