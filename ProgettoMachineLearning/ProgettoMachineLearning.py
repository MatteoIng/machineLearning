from random import random
import pandas
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, KernelPCA


chiavi=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]


#carico file per il dataset
heart=pandas.read_csv(r"C:\Users\MATTEO\Desktop\Programmi\Python\ProgettoMachineLearning\heart.csv")

#inizializzazione x
x=heart.drop("target",axis=1)

#inizializzazione y con valore target
y=heart[:]["target"].values

#split in train e test di x e y in 80/20
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

print("x_train:\n",x_train)
print("x_test:\n",x_test)
print("y_train:\n",y_train)
print("y_test:\n",y_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#0------------------------------------------------------------------------------------------------------------
#Riduzione dimensionalità
informazione=1.0

while informazione > 0.50:

    pca = PCA()
    pca.fit(x_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= informazione) + 1

    print("Numero dimensionalita' usate {} con una conservazione del {}% dell'informazione".format(d,informazione*100))

    pca = PCA(n_components=d)
    x_reduced_pca = pca.fit_transform(x_train)
    x_reduced_test_pca=pca.fit_transform(x_test)


    #1--------------------------------------------------------------------------------------------------------
    #addestro modello lineare 
    print("MODELLO LINEARE")
    modelloLin=linear_model.LinearRegression()
    modelloLin.fit(x_reduced_pca,y_train)

    print("Accuracy score test linear reg:",modelloLin.score(x_reduced_test_pca,y_test))
    print("Accuracy score training linear reg:",modelloLin.score(x_reduced_pca,y_train))
    print("\n")


    #2--------------------------------------------------------------------------------------------------------
    #Addestro modello logistic
    print("LOGISTIC REGRESSION")
    valoriLogisticTest=[]
    c=0.1
    for i in range(4):

        modelloLogistic=linear_model.LogisticRegression(C=c)
        modelloLogistic.fit(x_reduced_pca,y_train)

        #valutazione 
        print("C:",c)
        val=modelloLogistic.score(x_reduced_test_pca,y_test)
        print("Accuracy score test logistic reg:",val)
        valoriLogisticTest.append(val)
        print("Accuracy score training logistic reg:",modelloLogistic.score(x_reduced_pca,y_train))
        print("\n")
        c=c*10

    #plt.scatter([0.1,1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1],valoriLogisticTest,color="black")
    #plt.plot([0.1,1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1],valoriLogisticTest,color="blue")
    #plt.xticks(())
    #plt.yticks(())
    #plt.show()




    #3--------------------------------------------------------------------------------------------------------
    #Addesto linear Reg con costo ridge
    print("LINEAR REGRESSION COSTO RIDGE")
    valoriRidgeTest=[]
    a=0.1
    for i in range(4):
        

        modelloRidge=linear_model.Ridge(alpha=a)
        modelloRidge.fit(x_reduced_pca,y_train)

        #valutazione 
        print("ALPHA",a)
        val=modelloRidge.score(x_reduced_test_pca,y_test)
        print("Accuracy score test ridge reg:",val)
        valoriRidgeTest.append(val)
        print("Accuracy score training ridge reg:",modelloRidge.score(x_reduced_pca,y_train))
        print("\n")
        a=a*10

    #plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],valoriRidgeTest,color="black")
    #plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],valoriRidgeTest,color="blue")
    #plt.xticks(())
    #plt.yticks(())
    #plt.show()


    #4--------------------------------------------------------------------------------------------------------
    #Addesto linear Reg con Lasso
    print("LINEAR REGRESSION LASSO")
    valoriLassoTest=[]
    a=0.1
    for i in range(4):

        modelloLasso=linear_model.Lasso(alpha=i)
        modelloLasso.fit(x_reduced_pca,y_train)

        #valutazione 
        print("ALPHA",a)
        val=modelloLasso.score(x_reduced_test_pca,y_test)
        print("Accuracy score test Lasso reg:",val)
        valoriLassoTest.append(val)
        print("Accuracy score training Lasso reg:",modelloLasso.score(x_reduced_pca,y_train))
        print("\n")
        a=a*10

    #plt.scatter([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],valoriLassoTest,color="black")
    #plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],valoriLassoTest,color="blue")
    #plt.xticks(())
    #plt.yticks(())
    #plt.show()


    #5--------------------------------------------------------------------------------------------------------
    #addestro modello SVC
    c=0.1
    valoriLassoTestL1=[]
    valoriLassoTestL2=[]
    for i in range(4):

        modelloSVCl2=LinearSVC(penalty='l2',C=c)
        modelloSVCl1=LinearSVC(penalty='l1',C=c)
        modelloSVCl2.fit(x_reduced_pca,y_train)
        modelloSVCl1.fit(x_reduced_pca,y_train)

        #valutazione 
        print("C",c)
        val2=modelloSVCl2.score(x_reduced_test_pca,y_test)
        val1=modelloSVCl1.score(x_reduced_test_pca,y_test)
        print("Accuracy score test Lasso reg L2:",val2)
        print("Accuracy score test Lasso reg L1:",val1)
        valoriLassoTestL2.append(val2)
        valoriLassoTestL1.append(val1)
        print("Accuracy score training Lasso reg L2:",modelloSVCl2.score(x_reduced_pca,y_train))
        print("Accuracy score training Lasso reg L1:",modelloSVCl1.score(x_reduced_pca,y_train))
        print("\n")
        c=c*10

    informazione-=0.10





