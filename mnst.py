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
        noTree=100
        critF='entropy'
        fileRF='randomforestall.ml'
        fileres='myresult.np'

        clf = ensemble.RandomForestClassifier(n_estimators=noTree,criterion=critF)
        print("Classifier:\n\t %s"%str(clf))
        clf=streamobj.loadMe(fileRF)
        print(test.shape)
        rfresult=clf.predict(test)
        print(rfresult.shape)
        df = pd.DataFrame(rfresult)
        df.index+=1
        df.to_csv('resultmnst.csv')
