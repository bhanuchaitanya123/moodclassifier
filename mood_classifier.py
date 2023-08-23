'''import tensorflow
import os
import cv2
data_path='C:/Users/munab/OneDrive/python/data'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels))
print(label_dict)
print(categories)
print(labels)
img_size=100
data=[]
target=[]
for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        try:
            resized=cv2.resize(img,(img_size,img_size))
            data.append(resized)
            target.append(label_dict[category])
        except:
            print("error")
import numpy as np

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,3))
target=np.array(target)
from keras.utils import to_categorical
#from keras.utils import np_utils
from np_utils import *
from tensorflow.keras.utils import to_categorical
new_target=to_categorical(target)
print(target.shape)
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense,Flatten,Activation,Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np
model=Sequential()
model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
#model.add(Dense(50,activation='relu',input_shape=(4,)))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
from sklearn.model_selection import train_test_split
train_data,test_data,train_target,test_target=train_test_split(data,new_target,test_size=0.1)
train_target.shape
print(train_data)
print(train_target)
hist=model.fit(train_data,train_target,epochs=100,verbose=1,validation_split=0.2)
from tensorflow.keras.models import load_model
import os
model.save(os.path.join('models','happy_sadmodel.h5'))'''
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from turtle import *
import eyed3
import os
import cv2
import time
import numpy as np
img_size=256
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    print('cheese..')
    time.sleep(2)
    img_p="C://Users//munab//OneDrive//python//Image//imag.jpg"
    cv2.imshow('frame',frame)
    cv2.imwrite(img_p,frame)
    break
cap.release()
cv2.destroyAllWindows()
dic={ 0 :'happy', 1 :'sad'}
model=load_model(os.path.join('models','happy_sadmodel.h5'))
print(model.make_predict_function())
img_path="C://Users//munab//OneDrive//python//Image//imag.jpg"
#img_path="C:\\Users\\munab\\OneDrive\\python\\static\\Image\\7cd10e0c-26cc-11ee-8602-f5f9b1551cc3.jpg"
img=cv2.imread(img_path)
i=image.load_img(img_path,target_size=(100,100))
i=image.img_to_array(i)/255.0
i=i.reshape(1,100,100,3)

p=np.argmax(model.predict(i),axis=1)
print(p)
#p=model.predict_classes(i)
print(dic[p[0]])
if dic[p[0]] == 'happy':
    for im in os.listdir(os.path.join('mood','happy_f')):
      print(im)
      duration=eyed3.load(im).info.time_secs
      os.system("start {}".format(im))
      time.sleep(duration)
    print('hello happy')
else:
    for i in os.listdir(os.path.join('mood','sad_f')):
      print(i)
      duration=eyed3.load(i).info.time_secs
      os.system("start {}".format(i))
      time.sleep(duration)
    print('hello sad')
    
