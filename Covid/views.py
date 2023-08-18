from django.shortcuts import render,redirect
from .forms import PhotoSubmitForm
from .models import Photo
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess


def analyse(path):
    xception = settings.XCEPTION
    idx_to_class = {0:'COVID', 1:'Lung_Opacity',2: 'Normal', 3:'Viral Pneumonia'}

    img = image.load_img(path,target_size=(224,224,3))
    img_ = image.img_to_array(img)
    img_ = np.expand_dims(img_,0)

    #xception 
    a = xception_preprocess(img_)
    a_ = xception.predict(a) * 100
    a_idx = int(np.argmax(a_,axis=1))
    return a_idx


def home(request):
    return render(request,'Covid/base.html')

def predict(request):
    if request.method=="POST":
        form = PhotoSubmitForm(request.POST,request.FILES)
        if form.is_valid():
            img = request.FILES['image']
            fs = FileSystemStorage()
            path = fs.save(img.name,img)
            url = fs.url(path)
            #form.save()
            i = '.'+url
            label_id = analyse(i)
            if label_id==0:
                label='COVID'
            elif label_id==1:
                label='Lung Opacity'
            elif label_id==2:
                label='Normal'
            elif label_id==3:
                label='Viral Pneumonia'
            response = {
                'url':url,
                'label': label,
            }
            return render(request,'Covid/predict_response.html',response)
        else:
            form = PhotoSubmitForm()
            return render(request,'Covid/predict.html',{'form':form})
    else:
        form = PhotoSubmitForm()
        return render(request,'Covid/predict.html',{'form':form})


def author(request):
    return render(request,'Covid/author.html')