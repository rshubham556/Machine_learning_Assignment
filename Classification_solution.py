import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.pylab  as plt


# load the dataset
AH_data=pd.read_csv("UpdatedCSVFile.csv")
data_clean=AH_data.dropna()

         
#split into training and test sets
predictors=data_clean[['id','stateOrProvince','bathrooms','bedrooms','listPrice','livingArea','yearBuilt','propertySubType','listingCategory','numParkingSpaces','photoscount']]
targets=data_clean.grade
pred_train,pred_test,tar_train,tar_test=train_test_split(predictors, targets, test_size=.3)

classifier=DecisionTreeClassifier()
classifiers=classifier.fit(pred_train,tar_train)
predictions=classifier.predict(pred_test)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
DecisionTree=(100*(sklearn.metrics.accuracy_score(tar_test,predictions)))
print(DecisionTree)

print("Kneighbors classifier")
classifier=KNeighborsClassifier()
classifiers=classifier.fit(pred_train,tar_train)
predictions=classifier.predict(pred_test)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
#for accuracy
KNN=(100*(sklearn.metrics.accuracy_score(tar_test,predictions)))
print(KNN)

#KNeighbors model on training data
print("logisticRegression classifier")
classifier=LogisticRegression()
classifiers=classifier.fit(pred_train,tar_train)
predictions=classifier.predict(pred_test)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
#for accuracy
LR=(100*(sklearn.metrics.accuracy_score(tar_test,predictions)))
print(LR)



#***************************svm******************************
classifier=svm.SVC()
classifier=classifier.fit(pred_train,tar_train)
classifier.score(pred_train,tar_train)
predictions=classifier.predict(pred_test)
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
#for accuracy
SVM=(100*(sklearn.metrics.accuracy_score(tar_test,predictions)))
print(SVM)



#plot the graph
fig, ax = plt.subplots()

plt.bar([1],[DecisionTree],color='#DA0000',width=.35,label="DecisionTree")
plt.bar([2],[LR],color='#006400',width=.35,label="LogisticRegression")
plt.bar([3],[SVM],color='#0F00F0',width=.35,label="SVM")
plt.legend()
ax.set_xticklabels(('', 'DecisionTree','', 'LogisticRegression','','SVM'))
plt.ylabel('Accuracy')
plt.title('Comparision between DecisionTree, LogisticRegression, and  SVM')
plt.show()


fig, ax = plt.subplots()

plt.bar([1],[100-DecisionTree],color='#DA0000',width=.35,label="DecisionTree")
plt.bar([2],[100-LR],color='#006400',width=.35,label="LogisticRegression")
plt.bar([3],[100-SVM],color='#0F00F0',width=.35,label="SVM")
plt.ylim((0,100))
plt.legend()
ax.set_xticklabels(('', 'DecisionTree','', 'LogisticRegression','','SVM'))
plt.ylabel('Accuracy')
plt.title('Comparision of Miss Classification percentage of DecisionTree, LogisticRegression, and  SVM')
plt.show()


























