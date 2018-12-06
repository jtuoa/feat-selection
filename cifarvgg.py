
from __future__ import print_function
import keras
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import pdb
from sklearn.metrics import confusion_matrix
from utils import *

parser = argparse.ArgumentParser(description="cifar VGG")
parser.add_argument("--mode", type=str, default="train", help="train, test, extract")
parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10,100")
parser.add_argument("--out_dir", type=str, default="tmp", help="output directory to save models")
parser.add_argument("--chkpnt", type=str, default="", help="path to chkpnt. used in test and extract mode")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--nepoch", type=int, default=250, help="number of epoch")
parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")

args = parser.parse_args()

def ensure_dir(d):
    if len(d)  == 0: # for empty dirs (for compatibility with os.path.dirname("xxx.yy"))
        return
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as e:
            if e.errno != 17: # FILE EXISTS
                raise e

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, (layer,i)) for i,layer in enumerate(model.layers)])
    layer,idx = layer_dict[layer_name]
    return layer,idx

class cifarvgg:
    def __init__(self,train=True,num_classes = 10):
        self.num_classes = num_classes
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        if not train:
            K.set_learning_phase(0)

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights(args.chkpnt)


    def build_model(self):

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',input_shape=self.x_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1024,name='fc1'))
        model.add(BatchNormalization())
        model.add(Activation('relu',name='fc1_act'))

        model.add(Dense(1024,name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu',name='fc2_act'))

        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X,mode):
        if mode == "train":
            mean = np.mean(X,axis=(0,1,2,3))
            std = np.std(X, axis=(0, 1, 2, 3))
            np.save('mean_std_cifar%d.npy'%self.num_classes,(mean,std))
        else:#test
            mean,std = np.load('mean_std_cifar%d.npy'%self.num_classes)
        X = (X-mean)/(std+1e-7)
        return X

    def extract(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize(x,"test")
        model = self.model
        feat_layer = keras.models.Model(inputs=model.input,outputs=model.get_layer('fc2_act').output)
        return feat_layer.predict(x,batch_size)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize(x,"test")
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = args.batch_size
        maxepoches = args.nepoch
        learning_rate = args.lr
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        if args.dataset == "cifar10":
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        #TODO add validation split
        x_test = x_test.astype('float32')
        x_train = self.normalize(x_train,mode = "train")
        x_test = self.normalize(x_test,mode = "test")

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)


        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

        #checkpoints
        ensure_dir(args.out_dir)
        filepath= os.path.join(args.out_dir,"cifar%d-{epoch:02d}-{val_acc:.2f}.hdf5"%self.num_classes)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


        # training process in a for loop with learning rate drop every 25 epoches.
        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr,checkpoint],verbose=1)
        model.save_weights(os.path.join(args.out_dir,'cifar%dvgg.h5'%self.num_classes))
        return model

def visualize_samples(x_train,y_train):
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            if x_train[idx].shape[-1] == 3:
                plt.imshow(x_train[idx].squeeze().astype('uint8')) #change cmap='gray' when using gray input
            else:
                plt.imshow(x_train[idx].squeeze().astype('uint8'),cmap='gray')

            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.savefig('cifar10.png',bbox_inches='tight')
    plt.show()

def main():
    nclass = 0
    if args.dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        nclass = 10
    else:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        nclass = 100

    #visualize_samples(x_train,y_train)
    #return

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, nclass)
    y_test = keras.utils.to_categorical(y_test, nclass)
    if args.mode == "train":
        model = cifarvgg(num_classes=nclass)
    elif args.mode == "test":
        model = cifarvgg(train=False,num_classes=nclass)
        predicted_x = model.predict(x_test)
        acc = np.argmax(predicted_x,1)==np.argmax(y_test,1)
        '''
        test_y = y_test.argmax(1)
        predicted_x = predicted_x.argmax(1)
        cm = confusion_matrix(test_y,predicted_x)
        print(cm)
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        plot_cm(cm,classes)
        nclass = 10
        print("label    precision    recall")
        for label in range(nclass):
            print("%5d, %9.3f, %9.3f"%(label,precision(label, cm),recall(label, cm)))

        print("precision total:", precision_macro_average(cm))
        print("recall total:", recall_macro_average(cm))
        print("Accuracy:", ((sum((test_y==predicted_x))*1.)/test_y.shape[0])*100)
        print('Done')
        '''


        acc = sum(acc)/len(acc)
        print("the validation accuracy is: ",acc)
    else:# extract features
        model = cifarvgg(train=False,num_classes=nclass)
        out_dir = os.path.join(*args.chkpnt.split('/')[:-1])

        features_train = model.extract(x_train)
        data = {'data':features_train,'labels':y_train}
        np.save(os.path.join(out_dir,'train_feat.npy'),data)
        print('Done train data extraction ..')

        features_test = model.extract(x_test)
        data = {'data':features_test,'labels':y_test}
        np.save(os.path.join(out_dir,'test_feat.npy'),data)
        print('Done test data extraction ..')

    print('Done')
main()
