# Convolutional Neural Network

# Building the CNN

# Importing the required calsses from the keras library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# Initializing the CNN
classifier = Sequential()

#1. Convolution layer with ReLU 
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu" ))

#The first argument is the number of feature detectors(which in this case is 32)
#(3,3) is the dimensions of the feature detector (which is a matrix)
#input_shape is the shape of the image on which we apply the convolution operation
#All images will be in this new format of (64,64,3) which is a 3D array, 3 is the number of channels(for RGB)
#Black&White images are converted into 2D arrays (size,1) as only one channel is needed
#If working on GPU we can choose larger dimensions for the images
#The order of the input shape parameter mentioned in the element inspector is the order specified in the theano backend. 
#The order used here is for the tensorflow backend(dimensions,3/1)
#Activation function used is for introducing Non-Linearity 

#2. MaxPooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#pool_size is size of pooling matrix. (2,2) is highly recomended

#Adding another convolution layer with max pooling(Deeper Model)
classifier.add(Conv2D(32, (3, 3), activation="relu" ))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#here inpput shape is not required as the input to this layer is a feature map
#Number of feature detectors can be doubled each time. General strategy

#3. Flattening
classifier.add(Flatten())

#4. Full Connection
classifier.add(Dense(units = 128, activation='relu' ))

#we need to pic a large enough number in this case as we had 32 feature maps so will get many elements in the flattened vector
#general rule of thumb is to choose a number close to 100 which is a power of 2 hence 128. We could also use 100

#5. Output layer
classifier.add(Dense(units = 1, activation='sigmoid'))

#Units = 1 as we are expecting only one output. Either cat or dog
#If we had more than 2 categories, then activation = 'softmax'


# Compiling the CNN - Applying stochastic gradient descent 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#As binary outcome hence loss = 'binary_crossentropy' else use loss = 'categorical_crossentropy'

# Fitting the CNN to the Images

#Image augmentation - As image classifiers require a lot of data to be trained on, so we can either give it many images 
#or we can use data augmentation which will flip,tilt,swirl etc the image and train our model very well. It also avoids overfitting

#Importing ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

#This code section (borrowed from keras documentation).

#image augmentation for training set, we can use more transformations too
train_datagen = ImageDataGenerator(
        rescale=1./255,       #all image values will be between 0&1, rather than 0-255
        shear_range=0.2,      #applies random transvections    
        zoom_range=0.2,       #applies random zooms
        horizontal_flip=True) #flips horizontally

#image augmentation for test set        
test_datagen = ImageDataGenerator(rescale=1./255) #here we only rescale the pixel values

#Creating training set   
training_set = train_datagen.flow_from_directory(
        'dataset/training_set', #path of our file
        target_size=(64, 64),   #should be same as input_shape chose above
        batch_size=32,          #number of images after which weights will be updated
        class_mode='binary')    #indicates whether our output is binary or not (in this case binary)

#Creating test set
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#Fit model and test it
classifier.fit_generator(
        training_set,
        steps_per_epoch=(8000),     #total number of images in training set
        epochs=5,
        validation_data=test_set,
        validation_steps=(2000))    #total number of images in test set 

         





#In order to increase the accuracy of the CNN we can:-
#Add another classification layer
#Add another fully connected layer
