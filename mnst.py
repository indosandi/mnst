from glob import glob
from datetime import datetime
import numpy as np
import pandas as pd
from streamobj import streamobj
from sklearn import ensemble, cross_validation, preprocessing
from sklearn.decomposition import PCA



if __name__ == "__main__":
	start = datetime.now()
        ptrain=pd.read_csv('train.csv')
        ptest=pd.read_csv('test.csv')
        target=ptrain.values[:,0]
        train=ptrain.values[:,1:]
        test=ptest.values
        del ptrain
        del ptest
        
        print(test.shape)
        #noeig=200
        #pca=PCA(n_components=noeig)
        #pca.fit(train)
        #traindata=pca.fit_transform(train)
        #print(target.shape)
        #print(traindata.shape)
        noTree=100
        critF='entropy'
        fileRF='randomforestall.ml'
        fileres='myresult.np'

        clf = ensemble.RandomForestClassifier(n_estimators=noTree,criterion=critF)
        print("Classifier:\n\t %s"%str(clf))
        #x_train, x_test, y_train, y_test = cross_validation.train_test_split(traindata, target, test_size=0.2, random_state=1)
        #clf=clf.fit(train,target)
        #print(clf.score(x_test,y_test))
        #streamobj.saveMe(clf,fileRF)
        clf=streamobj.loadMe(fileRF)
        print(test.shape)
        rfresult=clf.predict(test)
        print(rfresult.shape)
        #result = [[0 for x in range(28001)] for x in range(2)] 

        #result[0][0]='Label'
        #result[0][1]='ImageId'
        #result[1:28001][0]=range(1,28001)
        #result[1:28001][1]=rfresult[0:28000]
        #result=np.ones((28000,2))  
        #result[:,0]=np.arange(0,28000)
        #result[:,1]=rfresult[:]
        df = pd.DataFrame(rfresult)
        df.index+=1
        df.to_csv('resultmnst.csv')
        #df.to_csv('resultmnst.csv',index=True,index_label=range(2,28001))
        #np.savetxt("resultmnst.csv", result, delimiter=",",fmt='%1d')
