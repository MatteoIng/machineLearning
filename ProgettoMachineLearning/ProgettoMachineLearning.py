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

#print("x_train:\n",x_train)
#print("x_test:\n",x_test)
#print("y_train:\n",y_train)
#print("y_test:\n",y_test)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

valoriLinearTest=[]
valoriLogisticTest=[]
valoriRidgeTest=[]
valoriLassoTest=[]
valoriSVCTestL1=[]
valoriSVCTestL2=[]

#0------------------------------------------------------------------------------------------------------------
#Riduzione dimensionalità
informazione=0.99
features=[]
featuresLinear=[]

while informazione > 0.75:

    pca = PCA()
    pca.fit(x_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= informazione) + 1

    featuresLinear.append(d)
    print("---------Numero dimensionalita' usate {} con una conservazione del {}% dell'informazione -------------\n".format(d,informazione*100))
    
    pca = PCA(n_components=d)
    x_reduced_pca = pca.fit_transform(x_train)
    x_reduced_test_pca=pca.fit_transform(x_test)

 

    #1--------------------------------------------------------------------------------------------------------
    #addestro modello lineare
    
    print("MODELLO LINEARE")
    modelloLin=linear_model.LinearRegression()
    modelloLin.fit(x_reduced_pca,y_train)

    val=modelloLin.score(x_reduced_test_pca,y_test)
    valoriLinearTest.append(val)
    print("Accuracy score test linear reg:",val)
    print("Accuracy score training linear reg:",modelloLin.score(x_reduced_pca,y_train))
    print("\n")


    #2--------------------------------------------------------------------------------------------------------
    #Addestro modello logistic
    print("LOGISTIC REGRESSION")
    
    c=0.1
    for i in range(4):
        features.append(d)

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

   


    #3--------------------------------------------------------------------------------------------------------
    #Addesto linear Reg con costo ridge
    print("LINEAR REGRESSION COSTO RIDGE")
    
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
        


    #4--------------------------------------------------------------------------------------------------------
    #Addesto linear Reg con Lasso
    print("LINEAR REGRESSION LASSO")
    
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

 

    #5--------------------------------------------------------------------------------------------------------
    #addestro modello SVC
    print("SVC")
    c=0.1
    
    for i in range(4):

        modelloSVCl2=LinearSVC(penalty='l2',C=c)
        modelloSVCl1=LinearSVC(penalty='l1',C=c)
        modelloSVCl2.fit(x_reduced_pca,y_train)
        #modelloSVCl1.fit(x_reduced_pca,y_train)

        #valutazione 
        print("C",c)
        val2=modelloSVCl2.score(x_reduced_test_pca,y_test)
        #val1=modelloSVCl1.score(x_reduced_test_pca,y_test)
        print("Accuracy score test SVC reg L2:",val2)
        #print("Accuracy score test SVC reg L1:",val1)
        valoriSVCTestL2.append(val2)
        #valoriSVCTestL1.append(val1)
        print("Accuracy score training SVC reg L2:",modelloSVCl2.score(x_reduced_pca,y_train))
        #print("Accuracy score training SVC reg L1:",modelloSVCl1.score(x_reduced_pca,y_train))
        print("\n")
        c=c*10

    informazione-=0.03
    print("\n")


plt.plot(featuresLinear,valoriLinearTest,color="orange",label="Logistic")
#plt.plot(features,valoriLogisticTest,color="blue",label="Logistic")
#plt.plot(features,valoriRidgeTest,color="black",label="Ridge")
#plt.plot(features,valoriLassoTest,color="green",label="Lasso")
#plt.plot([0.1,1,10,100],valoriSVCTestL1,color="red")
#plt.plot(features,valoriSVCTestL2,color="yellow",label="SVCL2")
plt.xlabel("n_features")
plt.ylabel("Value")
plt.legend(loc="lower right", title="Legend Title")

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_trisurf(features, valoriLogisticTest, [0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100], color='white', edgecolors='grey', alpha=0.5)
ax1.scatter(features, valoriLogisticTest, [0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100], c='red')
ax1.set_xlabel('n_features')                       
ax1.set_ylabel('valori_test_accurency')
ax1.set_zlabel('C_fitting')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_trisurf(features, valoriRidgeTest, [0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100], color='white', edgecolors='grey', alpha=0.5)
ax2.scatter(features, valoriRidgeTest, [0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100], c='red')
ax2.set_xlabel('n_features')                       
ax2.set_ylabel('valori_test_accurency')
ax2.set_zlabel('Alpha_fitting')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_trisurf(features, valoriLassoTest, [0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100], color='white', edgecolors='grey', alpha=0.5)
ax3.scatter(features, valoriLassoTest, [0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100], c='red')
ax3.set_xlabel('n_features')                       
ax3.set_ylabel('valori_test_accurency')
ax3.set_zlabel('Alpha_fitting')

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.plot_trisurf(features, valoriSVCTestL2, [0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100], color='white', edgecolors='grey', alpha=0.5)
ax4.scatter(features, valoriSVCTestL2, [0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100,0.1,1,10,100], c='red')
ax4.set_xlabel('n_features')                       
ax4.set_ylabel('valori_test_accurency')
ax4.set_zlabel('C_fitting')

plt.show()



