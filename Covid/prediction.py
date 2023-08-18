import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16  import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

xception = load_model('models/xception_model.h5')
inc_resv2 = load_model('models/InceptionResnet2_model.h5')
inception = load_model('models/Inception_model.h5')
resnet50v2 = load_model('models/Inception_model.h5')
densenet = load_model('models/Densenet_model.h5')
vgg16 = load_model('models/VGG_model.h5')
resnet50 = load_model('models/resnet50_model.h5')

path = input("Enter the File Path : ")
#C:\\Users\\HP\\Desktop\\Projects & Pratice\\Projects\\COVID-19_Radiography_Dataset\\Normal\\Normal-1.png

idx_to_class = {0:'COVID', 1:'Lung_Opacity',2: 'Normal', 3:'Viral Pneumonia'}
img = image.load_img(path,target_size=(224,224,3))
img_ = image.img_to_array(img)
plt.imshow(img)
plt.title(path.split('/')[2])
plt.show()
img_ = np.expand_dims(img_,0)

#xception 
a = xception_preprocess(img_)
a_ = xception.predict(a) * 100
a_idx = int(np.argmax(a_,axis=1))
print(f'Xception Model : {idx_to_class[a_idx]}')

#inception_resnet_v2
b = inception_resnet_preprocess(img_)
b_ = inc_resv2.predict(b) * 100
b_idx = int(np.argmax(b_,axis=1))
print(f'InceptionResnetV2 : {idx_to_class[b_idx]}')

#inception
c = inception_preprocess(img_)
c_ = inception.predict(c) * 100
c_idx = int(np.argmax(c_,axis=1))
print(f'InceptionV3 : {idx_to_class[c_idx]}')

#resnet50v2 && resnet
d = resnet50_preprocess(img_)
d_ = resnet50v2.predict(d) * 100
#d__ = resnet50.predict(d) * 100
d_idx = int(np.argmax(d_,axis=1))
print(f'Resnet50v2 : {idx_to_class[d_idx]}')
#d__idx = int(np.argmax(d__,axis=1))
#print(idx_to_class[d__idx])

#densenet
e = densenet_preprocess(img_)
e_ = densenet.predict(e) * 100
e_idx = int(np.argmax(e_,axis=1))
print(f'Densenet : {idx_to_class[e_idx]}')

#vgg16
f = vgg_preprocess(img_)
f_ = densenet.predict(f) * 100
f_idx = int(np.argmax(f_,axis=1))
print(f'VGG16 : {idx_to_class[f_idx]}')

avg = (a_ + b_ + c_ + d_ + e_ + f_)/6 
avg = avg.reshape((4,))
print(f'\nFINAL REPORT : ')
print(f'\t\tCOVID : {avg[0]:.2f}%')
print(f'\t\tLung_Opacity : {avg[1]:.2f}%')
print(f'\t\tNormal : {avg[2]:.2f}%')
print(f'\t\tViral Pneumonia : {avg[3]:.2f}%')

