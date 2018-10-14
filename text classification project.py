
# coding: utf-8

# In[1]:


#importing the os module and functions required for operating files
from os import listdir
from os.path import isfile, join


# In[2]:


#declaring a dictionary
dictionary2={}


# In[3]:


#creating the dictionary containing newgroups' names and the corresponding documents

folders = [f for f in listdir("C://Users//Admin//newsgroups//")]
for eachfolder in folders:
    folderloc="C://Users//Admin//newsgroups//" +eachfolder
    myfiles = [f for f in listdir(folderloc) if isfile(join(folderloc, f))]
    dictionary2[eachfolder]={}
    for eachfile in myfiles:
        filename=eachfile
        fileloc= folderloc+ "//"+filename
        file=open(fileloc,'r') 
        data= file.read()
        dictionary2[eachfolder][filename]=data.split()
        file.close() 


# In[4]:


#creating a list of tuples (text_document,category) out of dictionary 

mylist=[]
for key,value in dictionary2.items():
    category =key 
    for fname ,text in value.items():
        mylist.append((text,category))


# # Cleaning the documents

# In[5]:


# creating list of stopwords using nltk

from nltk.corpus import stopwords
import string 
#listing the punctuation marks
punctuations = list(string.punctuation)
#nltk stopwords donot include punctuation marks
stop= stopwords.words('english')
#complete list of stopwords including punctuations
stops= stop+punctuations


# In[6]:


#function for cleaning a single document 

def clean(document):
    output = []
    #iterating through every word in document
    for word in document :
        #checking if word is not in stops and hence adding it to the output document
        if not(word.lower() in stops) :
            clean_word= word 
            output.append(clean_word.lower())
    return output     


# In[7]:


#storing cleaned documents

cleaned_documents=[]
for document,category in mylist:
    c_doc=clean(document)
    cleaned_documents.append((c_doc,category))


# # Preparing the dataset

# In[8]:


#importing countvectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[9]:


#preparing arguments for train_test_split function 
categories=[category for document, category in cleaned_documents] 
text_documents=[" ".join(document) for document,category in cleaned_documents] 


# In[10]:


#importing train_test_split 
from sklearn.model_selection import train_test_split 


# In[11]:


#splitting the dataset
x_train,x_test,y_train,y_test=train_test_split(text_documents,categories,shuffle=True) 


# In[17]:


#creating count vectorizer object 
count_vector =CountVectorizer(max_features=2500)


# In[18]:


#creating training data
x_train_feature= count_vector.fit_transform(x_train)
x_train_matrix= x_train_feature.todense()


# In[19]:


#creating testing data
x_test_feature= count_vector.transform(x_test)
x_test_matrix= x_test_feature.todense()


# # Text classification using sklearn MultiNomial Naive Bayes Classifier

# In[20]:


#import multinomial naive bayes classifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report ,confusion_matrix


# In[51]:


#predicting the documents 
clf= MultinomialNB()
clf.fit(x_train_matrix,y_train)
y_pred= clf.predict(x_test_matrix)


# # Text classification using self implemented naive bayes algorithm

# In[22]:


# importing numpy module 
import numpy as np


# In[23]:


# counting the number of features
features= count_vector.get_feature_names()
n_features= len(features)


# In[24]:


#converting training and testing data into numpy arrays
x_train=np.array(x_train_matrix)
y_train=np.array(y_train)
x_test=np.array(x_test_matrix)
y_test=np.array(y_test)


# In[25]:


#checking the shape of training and testing data
x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[26]:


#fit function
#creating a dictionary
def fit(x_train,y_train):
    
    #result contains class_names and 'total_no_of_documents' as keys
    result={}
    #possible values of classes of newsgroups
    class_values= set(y_train)
    #storing count of total no. of documents
    result["total_no_of_documents"]=len(y_train)
    
    #iterating through every class in classes
    for current_class in class_values :
        #every class key is again a dictionary storing the features and total count of documents of that class
        result[current_class]={}
        #initialising the total words of a particular classs to 0
        result[current_class]["total_feature_points"]=0
        current_class_rows=(y_train==current_class)
        x_train_current=x_train[current_class_rows]
        y_train_current=y_train[current_class_rows]
        #storing total count of documents of current class
        result[current_class]["total_count"]=len(y_train_current)
        
        #iterating through each feature and storing the no. of words belonging to that feature
        for i in range(1,n_features+1):
            result[current_class][i]=(x_train_current[:,i-1]).sum()
            result[current_class]["total_feature_points"] += result[current_class][i]
    return result       


# In[29]:


#prediction for whole x_test
def predict(x_test,dictionary):
    
    y_pred=[]
    
    #iterating through every data point in x_test
    for point in x_test:
        #predicting class for a single data point
        point_class=predict_single_point(point,dictionary)
        #appending the class to y_pred 
        y_pred.append(point_class)
        i=i+1
    return y_pred    


# In[35]:


#prediction for a single point in x_test
def predict_single_point(point,dictionary):

    classes= dictionary.keys()
    #arbitarily assigning the values of best probability and best class
    best_p=-100000
    best_class=-1
    
    #assigning first_run to True to ensure the values of best probability and best class change irrespective of their initial values
    first_run=True
    for current_class in classes:
        #skipping the key which is not a class
        if current_class=="total_no_of_documents":
            continue
        #calculating probability for current class
        p_current_class= probability(dictionary,point,current_class) 
        if  (first_run or p_current_class>best_p): 
            best_p= p_current_class 
            best_class= current_class 
        first_run =False    
    return best_class 


# In[46]:


#probability function
def probability(dictionary,point,current_class):
    
    # P(class =current_class)
    output = np.log(dictionary[current_class]["total_count"])-np.log(dictionary["total_no_of_documents"])
    
    for j in range( 1,n_features+1):
        pointj=point[j-1]
        #ignoring those words which have zero count
        if pointj==0:
            continue
        # P(feature[i]=x[i] and class=current_class)    
        p_feature= np.log(dictionary[current_class][j])-np.log(dictionary[current_class]["total_feature_points"])
        #adding the log probabilities for a particular class
        output+= p_feature
    return output    


# In[47]:


#fitting the data and predicting the results
result =fit(x_train,y_train)
y_pred_self =predict(x_test,result)


# # Comparing the results

# In[52]:


#comparing the results 

#results for sklearn multinomial naive bayes classifiers 
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

#results for self implemented naive bayes algorithm
print(classification_report(y_test,y_pred_self))
print(confusion_matrix(y_test,y_pred_self))

