#%%
import cv2
import os
import base64
import requests
import pickle
import numpy as np
# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#%%
url = "http://localhost:8080/api/gethog"

classAnimal = ['Cat','Dog']

def img2HOG(img):
    v, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer)
    data = "image data,"+str.split(str(img_str),"'")[1]
    response = requests.get(url, json={"img":data})
    
    return response.json()

def readData(path):
    response = []
    for sub in os.listdir(path):
        for fn in os.listdir(path + '/' + sub):
            img_file_name = os.path.join(path, sub, fn)
            try:
                img = cv2.imread(img_file_name)
                if img is not None:
                    res = img2HOG(img)
                    hog = list(res["hog"])
                    hog.append(classAnimal.index(sub))
                    response.append(hog)
                else:
                    print(f"Failed to load image: {img_file_name}")
            except Exception as e:
                print(f"Error processing image: {str(e)}")
    return response

def savePkl(filename ,path):
    animal = readData(path)
        
    write_path = filename + ".pkl"
    pickle.dump(animal, open(write_path,"wb"))
    print("data preparation is done")
 
def loadPkl(filename):
    dataset = pickle.load(open(filename + '.pkl','rb'))
    return dataset
#%%
train_dir = r'Dataset\train'
savePkl('train_animal',train_dir)
#%%
test_dir = r'Dataset\test'
savePkl('test_animal',test_dir)

#%%
dataset_train = loadPkl('train_animal')
print("Data train : ",len(dataset_train))
dataset_test = loadPkl('test_animal')
print("Data test : ",len(dataset_test))
# %%
train_arr = np.array(dataset_train)
x_train = train_arr[:,0:-1]
y_train = train_arr[:,-1]
# %%
test_arr = np.array(dataset_test)
x_test = test_arr[:,0:-1]
y_test = test_arr[:,-1]


#%%
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
# %%
y_pred = clf.predict(x_test)
print("Accuracy:",accuracy_score(y_test, y_pred)*100)
print("Confusion Matrix : \n",confusion_matrix(y_test,y_pred))
# %%
path_model ='Cat_Dog_Model.pkl'
pickle.dump(clf, open(path_model,'wb'))
# %%
