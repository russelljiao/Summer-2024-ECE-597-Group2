import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier

# Bayes Classifier
def bayes_classifier(train_data, train_labels, test_data, test_labels):
    model = GaussianNB()
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

# Support Vector Machine (SVM)
def svm_classifier(train_data, train_labels, test_data, test_labels):
    model = SVC(kernel='linear', probability=True)
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

# K-Nearest Neighbors Classifier
def knn_classifier(train_data, train_labels, test_data, test_labels, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

# Neural Network (underwork)
def neural_network_classifier(train_data, train_labels, test_data, test_labels):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=1)
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

# Random Forest Classifier (underwork)
def random_forest_classifier(train_data, train_labels, test_data, test_labels):
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

# Logistic Regression Classifier
def logistic_regression_classifier(train_data, train_labels, test_data, test_labels):
    model = LogisticRegression(solver='liblinear', random_state=1)
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

# Gradient Boosting Classifier
def gradient_boosting_classifier(train_data, train_labels, test_data, test_labels):
    model = GradientBoostingClassifier(n_estimators=100, random_state=1)
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    return accuracy, report

# Example usage:
#train_data = np.random.rand(100, 10)  
#train_labels = np.random.randint(0, 2, 100)  
#test_data = np.random.rand(20, 10)  
#test_labels = np.random.randint(0, 2, 20) 

# Bayes Classifier
accuracy, report = bayes_classifier(train_data, train_labels, test_data, test_labels)
print("Bayes Classifier Accuracy:", accuracy)
print("Bayes Classifier Report:\n", report)

# SVM Classifier
accuracy, report = svm_classifier(train_data, train_labels, test_data, test_labels)
print("SVM Classifier Accuracy:", accuracy)
print("SVM Classifier Report:\n", report)

# KNN Classifier
accuracy, report = knn_classifier(train_data, train_labels, test_data, test_labels)
print("KNN Classifier Accuracy:", accuracy)
print("KNN Classifier Report:\n", report)

# Neural Network Classifier
accuracy, report = neural_network_classifier(train_data, train_labels, test_data, test_labels)
print("Neural Network Classifier Accuracy:", accuracy)
print("Neural Network Classifier Report:\n", report)

# Random Forest Classifier
accuracy, report = random_forest_classifier(train_data, train_labels, test_data, test_labels)
print("Random Forest Classifier Accuracy:", accuracy)
print("Random Forest Classifier Report:\n", report)

# Logistic Regression Classifier
accuracy, report = logistic_regression_classifier(train_data, train_labels, test_data, test_labels)
print("Logistic Regression Classifier Accuracy:", accuracy)
print("Logistic Regression Classifier Report:\n", report)

# Gradient Boosting Classifier
accuracy, report = gradient_boosting_classifier(train_data, train_labels, test_data, test_labels)
print("Gradient Boosting Classifier Accuracy:", accuracy)
print("Gradient Boosting Classifier Report:\n", report)
