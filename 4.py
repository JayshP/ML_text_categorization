import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
def extract_features(t,tt,filename,tfile,yfile,ytest):
  i=0
  a=[]
  b=[]
  f = open(filename,'r')
  g=open(tfile,'r')
  h=open(yfile,'r')
  h1=open(ytest,'r')
  while i<2:
    j=f.readline()
    j1=g.readline()
    p=int(j)
    a.append(p)
    p=int(j1)
    b.append(p)
    i=i+1
  D=a[0]
  W=a[1]
  D1=b[0]
  W1=b[1]
  print (D,W)
  f.close()
  g.close()
  i=1
  #while i<D:
  #t1=t.loc[t['doc_id'] == 1]
  #t2=t1.loc[t1['word_id']==76]
  #t3=t1.loc[t1['word_id']==77]
  #print(t2,'\n',t3)
  arr=[]
  while i<=D:
    t1=t.loc[t['doc_id'] == i]
    j=1
    ar=[]
    while j<=W:
      t2=t1.ix[t1['word_id']==j,'freq']
      if len(t2)==0:
         s=0;
      else:
         t3=t2.index[0]
         s=t1.ix[t3,'freq']
      ar.append(s)
      j=j+1
    print('\n\n',len(ar))
    arr.append(ar)
    i=i+1
  i=1
  X=arr[:]
  
  sr=[]
  while i<=D1:
    t1=tt.loc[t['doc_id'] == i]
    j=1
    ar=[]
    while j<=W1:
      t2=t1.ix[t1['word_id']==j,'freq']
      if len(t2)==0:
         s=0;
      else:
         t3=t2.index[0]
         s=t1.ix[t3,'freq']
      ar.append(s)
      j=j+1
    print('\n\n',len(ar))#,'\n',ar)
    sr.append(ar)
    i=i+1
  #X=arr[:]
  #print (i)
  X_test=sr[:]
  print (X_test)
  i=0
  Y=[]
  while i<D:
    j=h.readline()
    p=int(j)
    Y.append(p)
    i=i+1
  i=0
  Y_test=[]
  while i<D1:
    j=h1.readline()
    p=int(j)
    Y_test.append(p)
    i=i+1


  #Y_test1=[]
  modelkmeans = KMeans(n_clusters=2 )
  modelkmeans.fit(X)
  #Y_test=[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
  #Y=[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
  #print (Y)
  k=modelkmeans.predict(X_test)
  print (k)
  print ("Accuracy is ", accuracy_score(Y_test,k)*100,"% for 2 clusters")
  n=8
  neigh = KNeighborsClassifier(n_neighbors = n)
  neigh.fit(X, Y) 
  y_pred = neigh.predict(X_test)
  print (y_pred)
  print (Y_test)
  print ("Accuracy is ", accuracy_score(Y_test,y_pred)*100,"% for K-Value:",n)
  
def main():
  if len(sys.argv) != 5:
    print ('usage: python3 1.py  inputfile testfile y_file Y_test_file')
    sys.exit(1)
  ifilename = sys.argv[1]
  tfile = sys.argv[2]
  yfile=sys.argv[3]
  ytest=sys.argv[4]
  cols=['doc_id','word_id','freq']
  t=pd.read_table(ifilename,sep=' ',header=None,names=cols,skiprows=3)
  t1=pd.read_table(tfile,sep=' ',header=None,names=cols,skiprows=3)
  #t1=t.loc[t['word_id'] == 2]
  #t1=t.ix[1,'freq']
  #t1=t1+2
  #t1=t.ix[0:3,'word_id':'freq']
  print(t1)
  extract_features(t,t1,ifilename,tfile,yfile,ytest)
  

if __name__ == '__main__':
  main()
