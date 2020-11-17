from django.shortcuts import render
from django.http import HttpResponseRedirect , HttpResponse
from django.views import View
from django.core.files.storage import FileSystemStorage
import numpy as np
import pandas as pd
from django.conf import settings
import cv2 as cv
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import Graph
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('./models/resnet.h5')






class WelcomeView(View) :
    def get(self,request , *args , **kwargs):
        return HttpResponse('yessssssssssssssssss')



class MainView (View) :
    def get(self ,request , *args , **kwargs):
        context = {}
        return render(request , r'car_brand_classifier/main.html' )
    def post(self , request , *args , **kwargs):
        uploaded_file = request.FILES['doc']

        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name , uploaded_file)
        url = fs.url(name)
        print(url)
        path = 'car_brand_classification/media/83488633_2670100673235439_7626360689187094528_n.jpg'
        img = image.load_img('.'+url, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img / 255
        img = img.reshape(1, 224, 224, 3)
        pred = model.predict(img)
        pred = np.argmax(pred)
        if pred == 0 :
            car = 'audi'
        elif pred == 1 :
            car = 'lamborghini'
        else :
            car = 'marcades'
        context = {'file': uploaded_file, 'url': url , 'pred' : car}


        return render(request , 'car_brand_classifier/main.html', context)

