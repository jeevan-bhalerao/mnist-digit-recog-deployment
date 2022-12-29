import shutil
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from keras.models import load_model
import os
from os import listdir
import cv2

app = FastAPI()


@app.post("/upload-file/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
       
    file_location = f"files/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(uploaded_file.file, file_object)

     
    model = load_model('MNIST_model.h5')

    folder_dir = os.getcwd() + "/files"
    #print(folder_dir)
    for images in os.listdir(folder_dir):
        img = cv2.imread(os.path.join(folder_dir,images))

        #print(img)

        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        input_image_resize = cv2.resize(grayscale, (28, 28))

        input_image_resize = input_image_resize/255


        image_reshaped = np.reshape(input_image_resize, [1,28,28,1])

        input_prediction = model.predict(image_reshaped)

        input_pred_label = np.argmax(input_prediction)

        #print(type(input_pred_label))
        #print('The Handwritten Digit is recognised as ', input_pred_label)

        return {"The Handwritten Digit is recognised as ":input_pred_label.tolist()}



# if __name__ == '__main__':

#     uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)


