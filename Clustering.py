from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy
import random
from sklearn.decomposition import TruncatedSVD
from pandas import DataFrame

vectorizer = TfidfVectorizer(stop_words='english')
df = pd.read_csv("train_set.csv",sep="\t")
my_content = df['Content']
my_category = df['Category']
X = vectorizer.fit_transform(my_content)
svd = TruncatedSVD(n_components=200, random_state=42)
tempVect = svd.fit_transform(X)

myVector = [[] for i in range(len(my_content))]

for i in range(len(my_content)) :
    myVector[i] = numpy.reshape(tempVect[i] , (1 ,200))

def kmeans(myvec, clength, k , mycat):
    centers = []
    for i in range(k) :
        temp = random.randint(0, clength)
        centers.append(myvec[temp])
    belongto = [[] for i in range(k)]
    prev_centers = [[] for i in range(k)]
    categories = [[] for i in range(k)]
    mySim = []
    iterations = 0
    maxiters = 20
    flag = True
    while (flag ==True):
        print("Counter = " , iterations)
        print()
        if (iterations > maxiters):
            flag = False
        else:
            if (numpy.array_equal(prev_centers, centers)):
                flag = False
            else:
                iterations += 1
                belongto = [[] for i in range(k)]
                categories = [[] for i in range(k)]
                for i in range(0, clength) : 
                    mySim = []
                    for j in range(0, k):
                        mySim.append(cosine_similarity(myvec[i], centers[j]))
                        #print("MySim : ", mySim)
                    maxIndex = mySim.index(max(mySim))
                    #print("MaxIndex" , maxIndex , " , belongto length" , len(belongto))
                    belongto[maxIndex].append(myvec[i])
                    categories[maxIndex].append(mycat[i])
                index = 0
                for b in belongto:
                    prev_centers[index] = centers[index]
                    centers[index] = numpy.mean(b, axis=0).tolist()
                    index += 1  
    return categories


z = kmeans(myVector, len(my_content), 5 , my_category)


tech_pd = []
film_pd = []
busin_pd = []
foot_pd = []
polit_pd = []

for i in range(len(z)) :
    tech = 0
    film = 0
    busin = 0
    foot = 0
    polit = 0
    for j in range(len(z[i])):
        if(z[i][j] == 'Technology'):
            tech += 1
        if(z[i][j] == 'Film'):
            film += 1
        if(z[i][j] == 'Business'):
            busin += 1
        if(z[i][j] == 'Football'):
            foot += 1
        if(z[i][j] == 'Politics'):
            polit += 1
    tech_pd.append(tech / len(z[i]))
    film_pd.append(film / len(z[i]))
    busin_pd.append(busin / len(z[i]))
    foot_pd.append(foot / len(z[i]))
    polit_pd.append(polit / len(z[i]))
    
df3 = DataFrame({'Politics':polit_pd , 'Business':busin_pd ,'Football':foot_pd , 'Film':film_pd ,
                'Technology':tech_pd} , index=['Cluster1' , 'Cluster2' , 'Cluster3' ,
                                                'Cluster4' , 'Cluster5'])

df3.to_csv("clustering_KMeans.csv", sep='\t')