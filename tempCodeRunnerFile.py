
# use the file path where your dataset is stored
data_path = 'UTKFace'
img_files = os.listdir(data_path)
len(img_files)

shuffle(img_files)
gender = [i.split('_')[1] for i in img_files]

target = []
for i in gender:
    i = int(i)
    target.append(i)
    data = []
img_size = 32
for img_name in img_files:
    img_path = os.path.join(data_path, img_name)
    img = cv2.imread(img_path)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))
        data.append(resized)
    except Exception as e:
        print("Exception: ", e)

        # data values are normalized
data = np.array(data)/255.0

# Reshaping of data
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))

new_target = to_categorical(target, num_classes=2)

# saving the file
np.save('target', new_target)
np.save('data', data)


# build Convolutional neural network leyar
model = Sequential()
model.add(Conv2D(100, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(100, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(100, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(400, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Split the data
train_data, test_data, train_target, test_target = train_test_split(
    data, target, test_size=0.1)
