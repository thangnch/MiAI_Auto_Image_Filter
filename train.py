from keras.models import Model
from keras.applications import MobileNetV2
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import pathlib
import numpy as np

input_size = (224, 224, 3)

def get_generator():
    # Đường dẫn đến folder ảnh
    data_dir = pathlib.Path('./data')
    
    # Tên class lấy bằng đúng tên thư mục (indoor, outdoor)
    class_names = np.array([folder.name for folder in data_dir.glob('*') if folder.name != ".DS_Store"])

    # Tạo ra một image_gen, có thực hiện rescale
    image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)

    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                        classes=list(class_names), target_size=(input_size[0], input_size[1]),
                                                        shuffle=True, subset="training")

    test_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                        classes=list(class_names), target_size=(input_size[0], input_size[1]),
                                                        shuffle=True, subset="validation")

    return train_data_gen, test_data_gen, class_names

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

    # In cấu trúc mạng
    model.summary()
    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

# Định nghĩa batch_size và epochs
batch_size = 32
epochs = 1

# Load các data gen
train_generator, validation_generator, class_names = get_generator()

# Tạo model
model = get_model(num_classes = len(class_names))

# Tạo callback để lấy weight mới nhất
checkpoint = ModelCheckpoint("models/my_model-" + "-loss-{val_loss:.2f}-acc-{val_accuracy:.2f}.h5", save_best_only=True, verbose=1)


# Train model
training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)
model.fit_generator(train_generator, steps_per_epoch=training_steps_per_epoch,
                    validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
                    epochs=epochs, verbose=1, callbacks=[ checkpoint])

# Lưu model sau khi train xong all epochs
model.save("models/my_model.h5")

