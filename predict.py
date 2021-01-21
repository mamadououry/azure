
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class dogcat:
    def __init__(self,filename):
        self.filename = filename


    def predictiondogcat(self):
        # load model
        model = load_model('modellessai.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        classes = ['ndeye', 'neo', 'oury']
        for i in range(0, 3):
            if result[0][i] == 1:
                prediction = classes[i]
                return [{"image": prediction}]


