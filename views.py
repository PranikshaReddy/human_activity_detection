from django.shortcuts import render
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from django.http import HttpResponse
import os
from .models import Actions
from .prediction import livestreaming


# Create your views here.

homepage = 'index.html'
resultpage = 'result.html'


Action = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 'fighting', 'hugging',
          'laughing', 'listening_to_music', 'running', 'sitting', 'sleeping', 'texting', 'using_laptop']


def index(request):
    return render(request, homepage)


def result(request):
    if request.method == 'POST':
        m = int(request.POST['alg'])
        file = request.FILES['file']
        fn = Actions(images=file)
        fn.save()
        path = os.path.join('webapp/static/images/', fn.filename())
        acc = pd.read_csv("webapp/Accurary.csv")

        if m == 1:
            new_model = load_model("webapp/models/CNN.h5", compile=False)
            test_image = image.load_img(path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image /= 255
            a = acc.iloc[m - 1, 1]

        else:
            new_model = load_model("webapp/models/MobileNet.h5", compile=False)
            test_image = image.load_img(path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image /= 255
            a = acc.iloc[m-1, 1]

        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        pred = Action[np.argmax(result)]
        print(pred)

        return render(request, resultpage, {'text': pred, 'path': 'static/images/'+fn.filename(), 'a': round(a*100, 3)})

    return render(request, resultpage)


def prediction(request):
    livestreaming()
    print("Running")
    return render(request, homepage)
