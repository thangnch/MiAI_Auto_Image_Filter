from keras.models import Model, load_model
from keras.applications import MobileNetV2
from keras.layers import Dense
import numpy as np
import cv2
import filters

input_size = (224, 224, 3)

def get_model(input_shape=input_size, num_classes = 2):
    # Load MobileNetV2
    mobilenet_model = MobileNetV2(input_shape=input_shape)
    # Bỏ đi layer cuối cùng (FC)
    mobilenet_model.layers.pop()
    # Đóng băng các layer (trừ 4 layer cuối)
    for layer in mobilenet_model.layers[:-4]:
        layer.trainable = False

    mobilenet_output = mobilenet_model.layers[-1].output

    # Tạo các layer mới
    output = Dense(num_classes, activation="softmax")
    # Lấy input từ output của MobileNet
    output = output(mobilenet_output)

    # Tạo model với input của MobileNet và output là lớp Dense vừa thêm
    model = Model(inputs=mobilenet_model.inputs, outputs=output)

    return model

# Thay đổi ảnh ở đây
image_path = "test_data/a.png"

# Đọc ảnh
image = cv2.imread(image_path)
image_org = image.copy()

# Chuyển đổi thành tensor
image = cv2.resize(image, dsize=input_size[:2])
image = image/255
image = np.expand_dims(image, axis=0)

# Tạo model
model = get_model()

# load the optimal weights
model.load_weights("models/my_model--loss-0.41-acc-0.94.h5")

# Tiến hành predict
class_names = ['indoor','landscape']
output = model.predict(image)
class_name = class_names[np.argmax(output)]

# Nếu là landscape thì apply overlay
if class_name == "landscape":
    filter_image = filters.apply_color_overlay(image_org, intensity=.2, red=250, green=100, blue=0)
    cv2.putText(filter_image,class_name,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
else:
    # Nếu là indoor thì apply sepia
    filter_image = filters.apply_sepia(image_org, intensity=.8)
    cv2.putText(filter_image, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('orginal_color', image_org)
cv2.imshow('color_overlay', filter_image)
cv2.waitKey()
cv2.destroyAllWindows()






