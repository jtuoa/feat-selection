from __future__ import division  # floating point division
import numpy as np
from sklearn.feature_selection import chi2
import keras
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.models import Sequential
from keras import regularizers
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #disable warnings and debug info from TF
import time
import utils
import pdb

class FeatureSelector:
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {'nselected':1000,'nbatch':32}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def select(self, Xtrain, ytrain):
        """ select subfeatures using the traindata """
        nselected = self.params['nselected']
        self.subfeat = np.random.randint(Xtrain.shape[1], size=nselected)

    def train(self, Xtrain, ytrain,Xvalid,yvalid):
        """ train subfeatures using the traindata """
        x = Xtrain[:,self.subfeat]
        xvalid = Xvalid[:,self.subfeat]
        nclass = np.unique(ytrain).shape[0]
        y = np_utils.to_categorical(ytrain, nclass)
        yvalid = np_utils.to_categorical(yvalid, nclass)

        #To generate the same weights initialization for all methods, keras uses numpy and tf as backend so we need to set tf's seed as well
        from tensorflow import set_random_seed
        np.random.seed(42)
        set_random_seed(2)

        self.model = Sequential()
        self.model.add(Dense(nclass, input_dim=x.shape[1], activation='softmax'))
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(x, y, batch_size=self.params['nbatch'], nb_epoch=20,verbose=1,validation_data=(xvalid,yvalid))

    def predict(self, Xtest):
        ytest= self.model.predict(Xtest[:,self.subfeat],50)
        return ytest.argmax(1)

    def plot_training():

        loss = self.history.history['loss']
        acc = self.history.history['acc']
        rng = range(len(self.loss))
        plt.plot(rng,self.loss)
        plt.plot(rng,self.acc)
        #validation
        loss = self.history.history['acc_loss']
        acc = self.history.history['acc_acc']
        plt.plot(rng,self.loss)
        plt.plot(rng,self.acc)

class FisherClass(FeatureSelector):
    """
    Fisher
    """
    def __init__( self, parameters={} ):
        self.params = {'nselected':1000,'nbatch':32,'Fisher_thresh':0.0}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.subfeat = None
        self.clf = None

    def select(self, Xtrain, ytrain):
        """ select subfeatures using the traindata """
        nselected = self.params['nselected']
        fisher_score,p_value = chi2(Xtrain, ytrain)
        NaNs = fisher_score!=fisher_score
        print("Number of nans %d: "%sum(NaNs))
        fisher_score[NaNs] = 0 #zero-out nans

        thresh = self.params['Fisher_thresh']
        if abs(thresh - 0.0) < 1e-8:
            f_score,idx = zip(*sorted(zip(fisher_score,range(Xtrain.shape[1])),reverse=True))
            self.subfeat = np.asarray(idx[:nselected])
        else:
            mask = fisher_score >= thresh
            idx = np.asarray(list(range(Xtrain.shape[1])))
            self.subfeat = idx[mask]
            if self.subfeat.shape[0] == 0: #if empty just select the highest feature to avoid pipeline crashing
                self.subfeat = np.asarray([np.argmax(fisher_score)])



class L1Class(FeatureSelector):
    """
    L1
    """
    def __init__( self, parameters={} ):
        self.params = {'nselected':1000,'nbatch':32,'L1_thresh':0.0, 'L1_regwgt':0.0}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.subfeat = None
        self.clf = None

    def select(self, Xtrain, ytrain):
        """ select subfeatures using the traindata """

        nclass = len(np.unique(ytrain))
        ytrain = np_utils.to_categorical(ytrain, nclass)

        model = Sequential()
        model.add(Dense(nclass, input_dim=Xtrain.shape[1], activation='softmax', kernel_regularizer=regularizers.l1(self.params['L1_regwgt'])))
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(Xtrain, ytrain, batch_size=self.params['nbatch'], nb_epoch=20,verbose=0)
        layer = model.layers[-1]
        weights = layer.get_weights()[0] #filter weights only (e.g weights[0]) not bias (e.g weights[1])

        thresh = self.params['L1_thresh']
        if abs(thresh - 0.0) < 1e-8:
            nselected = self.params['nselected']
            w,idx = zip(*sorted(zip(abs(weights),range(Xtrain.shape[1])),reverse=True))
            self.subfeat = idx[:nselected]
        else:
            l1_norm = np.sum(np.abs(weights),1)
            mask = l1_norm >= thresh
            idx = np.asarray(list(range(Xtrain.shape[1])))
            self.subfeat = idx[mask]
            if self.subfeat.shape[0] == 0: #if empty just select the highest feature to avoid pipeline crashing
                self.subfeat = np.asarray([np.argmax(l1_norm)])


        #self.subfeat = range(Xtrain.shape[1])

class MPClass(FeatureSelector):
    """
    MP
    """
    def __init__( self, parameters={} ):
        self.params = {'nselected':1000,'nbatch':32,'MP_eps':0.0}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.subfeat = None
        self.clf = None

    def select(self, Xtrain, ytrain):
        """ select subfeatures using the traindata """
        I = np.asarray([],dtype=np.int)
        epsilon = self.params['MP_eps']
        all_feat = np.asarray(list(range(Xtrain.shape[1])),dtype=np.int)
        nclass = len(np.unique(ytrain))
        prev_acc = -1<<26

        (trainx,trainy),(validx,validy) = utils.split_data(Xtrain,ytrain)
        trainy = np_utils.to_categorical(trainy, nclass)
        validy = np_utils.to_categorical(validy, nclass)
        for i in range(Xtrain.shape[1]):
            cur_set = np.asarray([a for a in all_feat if a not in I])
            acc = np.zeros_like(cur_set,dtype=np.float32)
            print('Currently selecting the %d-th feature with accuracy so far %f'%(I.shape[0],prev_acc))
            I_idx = np.zeros_like(I)
            I_idx = np.append(I_idx,0)
            start = time.time()
            for j,idx in enumerate(cur_set):
                I_idx[-1] = idx
                history = self.train_model(nclass,trainx[:,I_idx],trainy,validx[:,I_idx],validy)
                acc[j] = max(history.history['val_acc'])
                print('Feature %d with accuracy %f'%(cur_set[j],acc[j]))
                if j and j % 20 == 0: #as creating multiple models in a row causes slow down each iteration
                    K.clear_session() #https://github.com/keras-team/keras/issues/3579#issuecomment-242492693
            acc_sorted,idx = zip(*sorted(zip(acc,range(acc.shape[0])),reverse=True))
            new_feat = list(idx[:100])
            I = np.append(I,cur_set[new_feat])
            history = self.train_model(nclass,trainx[:,I],trainy,validx[:,I],validy,20)
            cur_acc = max(history.history['val_acc'])
            print('Current accuracy with %d features is %f '%(I.shape[0], cur_acc))
            end = time.time()
            print('This took %f seconds'%((end-start)))
            if (epsilon and abs(cur_acc - prev_acc) < epsilon) or (epsilon == 0.0 and I.shape[0] == self.params['nselected']):
                break;
            prev_acc = cur_acc

        self.subfeat = I

    def train_model(self,nclass,xtrain,ytrain,xvalid,yvalid,epoch=1):

        #To generate the same weights initialization for all methods, keras uses numpy and tf as backend so we need to set tf's seed as well
        from tensorflow import set_random_seed
        np.random.seed(2)
        set_random_seed(2)

        model = Sequential()
        model.add(Dense(nclass, input_dim=xtrain.shape[1], activation='softmax'))
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(xtrain, ytrain, batch_size=128, nb_epoch=epoch,verbose=0, validation_data=(xvalid,yvalid))
        return history

