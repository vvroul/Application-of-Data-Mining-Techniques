from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np 
from itertools import cycle
from pandas import DataFrame
from sklearn.cross_validation import StratifiedKFold


def create_roc_plot(clf , X_test , Y_test , le , algorithm_name , count) :
    y_score = clf.predict_proba(X_test)
    
    n_classes = 5
    fpr = dict()
    tpr = dict()
    roc_auc = dict()   

    Y_prob_final = []  
    # list of nd-arrays, the "i"-th nd-array holds the probability for each article to be of category "i" 
    fpr = []     #list of posibilities of False Positive Rate for every category 
    tpr = []
    for _pos_ in range(n_classes) :
        Y_prob_final.append( np.zeros(Y_test.shape[0]) )     # one ndarray for every category */
        for i in range( Y_test.shape[0] ) :      # for every article */
        # Y_prob_final[-1] : the nd-array I just appended */
            Y_prob_final[-1][i] = y_score[i][_pos_]      # store only the probability I am interested in */
        fpr1, tpr1, _ = metrics.roc_curve( Y_test, Y_prob_final[-1], pos_label=_pos_)
        fpr.append( fpr1 )
        tpr.append( tpr1 )

    # Plot all ROC curves
    fig = plt.figure()

    for i in range(n_classes):
        roc_auc[i] = auc(fpr[i], tpr[i])
        #print(roc_auc[i])
        
    #print(roc_auc)


    colors = cycle(['firebrick', 'midnightblue', 'fuchsia', 'aquamarine', 'gold'])
    for i, color in zip(range(n_classes), colors):
        categoryName = le.inverse_transform([i])[0]
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
            label='ROC curve of class ' + categoryName + ' (area = {1:0.2f})'.format(i, roc_auc[i]))



    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC curves for all 5 categories for roc ' + str(count))
    plt.legend(loc="lower right")
    fig.savefig( 'ROC_pngs\\' + algorithm_name + '_roc_10fold.png') # save in .png */
    plt.close(fig)
    
    return roc_auc
    
    

vectorizer = CountVectorizer(stop_words='english')
df = pd.read_csv("train_set.csv",sep="\t")
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y = le.transform(df["Category"])
X = vectorizer.fit_transform(df['Content'])

cv = StratifiedKFold(Y, n_folds=10)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.1, random_state=42)


print("MultinomialNB")
clf = MultinomialNB()
scores_multi = cross_val_score(clf, X, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_multi)
clf.fit(X_train,Y_train)
multi = clf.predict(X_test)

roc_business = []
roc_film = []
roc_football = []
roc_politics = []
roc_technology = []

for i ,(train, test) in enumerate(cv):
    classifier = MultinomialNB()
    classifier.fit(X[train], Y[train])
    name = "MultinomialNB_" + str(i)
    roc = create_roc_plot(classifier , X[test] , Y[test] , le , name , i)
    roc_business.append(roc[0])
    roc_film.append(roc[1])
    roc_football.append(roc[2])
    roc_politics.append(roc[3])
    roc_technology.append(roc[4])
    
    
roc_business = np.array(roc_business)
roc_business = roc_business.mean()
roc_film = np.array(roc_film)
roc_film = roc_film.mean()
roc_football = np.array(roc_football)
roc_football = roc_football.mean()
roc_politics = np.array(roc_politics)
roc_politics = roc_politics.mean()
roc_technology = np.array(roc_technology)
roc_technology = roc_technology.mean()


multi_pd = [scores_multi , roc_business , roc_film , roc_football ,roc_politics , roc_technology ]


#----------------------------------------------------------------------------------------------------------------

print("BernoulliNB")
clf2 = BernoulliNB()
scores_bernouli = cross_val_score(clf2, X, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_bernouli)
clf2.fit(X_train,Y_train)
bernouli = clf2.predict(X_test)

roc_business = []
roc_film = []
roc_football = []
roc_politics = []
roc_technology = []

for i ,(train, test) in enumerate(cv):
    classifier = BernoulliNB()
    classifier.fit(X[train], Y[train])
    name = "BernoulliNB_" + str(i)
    roc = create_roc_plot(classifier , X[test] , Y[test] , le , name , i)
    roc_business.append(roc[0])
    roc_film.append(roc[1])
    roc_football.append(roc[2])
    roc_politics.append(roc[3])
    roc_technology.append(roc[4])
    
    
roc_business = np.array(roc_business)
roc_business = roc_business.mean()
roc_film = np.array(roc_film)
roc_film = roc_film.mean()
roc_football = np.array(roc_football)
roc_football = roc_football.mean()
roc_politics = np.array(roc_politics)
roc_politics = roc_politics.mean()
roc_technology = np.array(roc_technology)
roc_technology = roc_technology.mean()


bernouli_pd = [scores_multi , roc_business , roc_film , roc_football ,roc_politics , roc_technology ]


#-----------------------------------------------------------------------------------------------------------


svd = TruncatedSVD(n_components=200, random_state=42)
X_lsi = svd.fit_transform(X)

X_train , X_test , Y_train , Y_test = train_test_split(X_lsi , Y , test_size=0.1, random_state=42)


#----------------------------------------------------------------------------------------------------------------

print("SVC")
clf3 = svm.SVC(C=1.0 , probability = True)
scores_svc = cross_val_score(clf3, X_lsi, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_svc)
clf3.fit(X_train,Y_train)
svc = clf3.predict(X_test)

roc_business = []
roc_film = []
roc_football = []
roc_politics = []
roc_technology = []

for i ,(train, test) in enumerate(cv):
    classifier = svm.SVC(C=1.0 , probability = True)
    classifier.fit(X_lsi[train], Y[train])
    name = "SVC_" + str(i)
    roc = create_roc_plot(classifier , X_lsi[test] , Y[test] , le , name , i)
    roc_business.append(roc[0])
    roc_film.append(roc[1])
    roc_football.append(roc[2])
    roc_politics.append(roc[3])
    roc_technology.append(roc[4])
    
    
roc_business = np.array(roc_business)
roc_business = roc_business.mean()
roc_film = np.array(roc_film)
roc_film = roc_film.mean()
roc_football = np.array(roc_football)
roc_football = roc_football.mean()
roc_politics = np.array(roc_politics)
roc_politics = roc_politics.mean()
roc_technology = np.array(roc_technology)
roc_technology = roc_technology.mean()


svc_pd = [scores_multi , roc_business , roc_film , roc_football ,roc_politics , roc_technology ]

    
#----------------------------------------------------------------------------------------------------------------
    
    
print("Random Forests")
clf4 = RandomForestClassifier(n_estimators=100)
scores_rf = cross_val_score(clf4, X_lsi, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_rf)
clf4.fit(X_train,Y_train)
rf = clf4.predict(X_test)

roc_business = []
roc_film = []
roc_football = []
roc_politics = []
roc_technology = []

for i ,(train, test) in enumerate(cv):
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_lsi[train], Y[train])
    name = "Random_Forest_" + str(i)
    roc = create_roc_plot(classifier , X_lsi[test] , Y[test] , le , name , i)
    roc_business.append(roc[0])
    roc_film.append(roc[1])
    roc_football.append(roc[2])
    roc_politics.append(roc[3])
    roc_technology.append(roc[4])
    
    
roc_business = np.array(roc_business)
roc_business = roc_business.mean()
roc_film = np.array(roc_film)
roc_film = roc_film.mean()
roc_football = np.array(roc_football)
roc_football = roc_football.mean()
roc_politics = np.array(roc_politics)
roc_politics = roc_politics.mean()
roc_technology = np.array(roc_technology)
roc_technology = roc_technology.mean()


rf_pd = [scores_multi , roc_business , roc_film , roc_football ,roc_politics , roc_technology ]
    
#----------------------------------------------------------------------------------------------------------------

    
print("KNearest Neighbor")
clf5 = KNeighborsClassifier(n_neighbors=5)
scores_knn = cross_val_score(clf5, X_lsi, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_knn)
clf5.fit(X_train,Y_train)
knn = clf5.predict(X_test)

roc_business = []
roc_film = []
roc_football = []
roc_politics = []
roc_technology = []

for i ,(train, test) in enumerate(cv):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_lsi[train], Y[train])
    name = "K-Nearest_Neighbor_" + str(i)
    roc = create_roc_plot(classifier , X_lsi[test] , Y[test] , le , name , i)
    roc_business.append(roc[0])
    roc_film.append(roc[1])
    roc_football.append(roc[2])
    roc_politics.append(roc[3])
    roc_technology.append(roc[4])
    
    
roc_business = np.array(roc_business)
roc_business = roc_business.mean()
roc_film = np.array(roc_film)
roc_film = roc_film.mean()
roc_football = np.array(roc_football)
roc_football = roc_football.mean()
roc_politics = np.array(roc_politics)
roc_politics = roc_politics.mean()
roc_technology = np.array(roc_technology)
roc_technology = roc_technology.mean()

knn_pd = [scores_knn , roc_business , roc_film , roc_football ,roc_politics , roc_technology ]


#------------------------------------------------------------------------------------------------------------------------

# Our method

k_range = range(1,20)
k_scores = []
for k in k_range :
    knn3 = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn3, X_lsi, Y , cv=10 , scoring='accuracy')
    k_scores.append(scores.mean())



maximun = max(k_scores)
maxIndex = k_scores.index(maximun)
maxIndex += 1
maxIndex

print("My Method")
clf6 = KNeighborsClassifier(n_neighbors=maxIndex)
scores_knn2 = cross_val_score(clf6, X_lsi, Y , cv=10 , scoring='accuracy').mean()
print("10-fold cross validation accuracy = " , scores_knn2)
clf6.fit(X_train,Y_train)
knn2 = clf6.predict(X_test)

roc_business = []
roc_film = []
roc_football = []
roc_politics = []
roc_technology = []

for i ,(train, test) in enumerate(cv):
    classifier = KNeighborsClassifier(n_neighbors=maxIndex)
    classifier.fit(X_lsi[train], Y[train])
    name = "My_Method_" + str(i)
    roc = create_roc_plot(classifier , X_lsi[test] , Y[test] , le , name , i)
    roc_business.append(roc[0])
    roc_film.append(roc[1])
    roc_football.append(roc[2])
    roc_politics.append(roc[3])
    roc_technology.append(roc[4])
    
    
roc_business = np.array(roc_business)
roc_business = roc_business.mean()
roc_film = np.array(roc_film)
roc_film = roc_film.mean()
roc_football = np.array(roc_football)
roc_football = roc_football.mean()
roc_politics = np.array(roc_politics)
roc_politics = roc_politics.mean()
roc_technology = np.array(roc_technology)
roc_technology = roc_technology.mean()

knn2_pd = [scores_knn2 , roc_business , roc_film , roc_football ,roc_politics , roc_technology ]

#----------------------------------------------------------------------------------------------------



count_multi = accuracy_score(Y_test,multi,normalize=False)
accuracy_multi = accuracy_score(Y_test,multi)
print("MultinomialNB made " , count_multi , " right predictions out of " , len(Y_test) , " , accuracy = " , accuracy_multi)

count_bernouli = accuracy_score(Y_test,bernouli,normalize=False)        
accuracy_bernouli = accuracy_score(Y_test,bernouli)
print("BernouliNB made " , count_bernouli , " right predictions out of " , len(Y_test) , " , accuracy = " , accuracy_bernouli)

count_svc = accuracy_score(Y_test,svc,normalize=False)        
accuracy_svc = accuracy_score(Y_test,svc)
print("SVC made " , count_svc , " right predictions out of " , len(Y_test) , " , accuracy = " , accuracy_svc)

count_rf = accuracy_score(Y_test,rf,normalize=False)        
accuracy_rf = accuracy_score(Y_test,rf)
print("Random Forests made " , count_rf , " right predictions out of " , len(Y_test) , " , accuracy = " , accuracy_rf)
    
count_knn = accuracy_score(Y_test,knn,normalize=False)
accuracy_knn = accuracy_score(Y_test,knn)
print("K-Nearest Neighbor made " , count_knn , " right predictions out of " , len(Y_test) , " , accuracy = " , accuracy_knn)

count_knn2 = accuracy_score(Y_test,knn2,normalize=False)
accuracy_knn2 = accuracy_score(Y_test,knn2)
print("My Method made " , count_knn2 , " right predictions out of " , len(Y_test) , " , accuracy = " , accuracy_knn2)





new_df = DataFrame({'MultinomialNB':multi_pd , 'BernouliNB':bernouli_pd ,
                    'SVC(SVM)':svc_pd , 'Random Forests':rf_pd ,
                    'K-nearest neighbors':knn_pd , 'My Method':knn2_pd} , index=['accuracy' , 'roc_business' , 'roc_film' ,
                                                                          'roc_football' , 'roc_politics' , 'roc_technology'])

new_df.to_csv("EvaluationMetric_10fold.csv", sep='\t')




# We create testSet_Categories.csv , only for multinomialNB

my_id = df['Id']
X_train = X[:10000]
Y_train = Y[:10000]
X_test = X[10000:]
Y_test = Y[10000:]


print("MultinomialNB")
clf = MultinomialNB()
clf.fit(X_train,Y_train)
multi = clf.predict(X_test)


temp_id = []
temp_cat = []

for i in range(len(Y_test)) :
    cat = ""
    if(multi[i] == 0) :
        cat = "Business"
    if(multi[i] == 1) :
        cat = "Film"
    if(multi[i] == 2) :
        cat = "Football"
    if(multi[i] == 3) :
        cat = "Politics"
    if(multi[i] == 4) :
        cat = "Technology"
    temp_id.append(my_id[i])
    temp_cat.append(cat)
                           
df2 = DataFrame({'Id':temp_id , 'Predicted category':temp_cat})

df2.to_csv("testSet_categories.csv", sep='\t')
