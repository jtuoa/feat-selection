import numpy as np
import argparse
import os
import algorithms as algs
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from scipy import stats
from utils import *
import pdb
import pickle

np.random.seed(42) #set seed for reproducibility

parser = argparse.ArgumentParser(description="Feature selection main")
parser.add_argument("--path", type=str, default="", help="path with train_feat.npy and test_feat.npy")
parser.add_argument("--fs", type=str, default="", help="feature selection method")
args = parser.parse_args()
nfeat = None #total number of features

def gen_kf(kf_dict):
    for n in range(len(kf_dict.keys())):
        yield kf_dict[n]

#add optional parameter, upload from kf_dict or not
def hyper_selection(x,y, kf_dictSelect):
    ntrial = 10
    K = 5
    '''
    'MP' : (algs.MPClass(), {
                            'MP_eps': np.random.uniform(low=1e-5, high=1, size=(ntrial,))
                            }),
    'Fisher' : (algs.FisherClass(), {
                                    'Fisher_thresh': np.random.uniform(low=1e-5, high=2e5, size=(ntrial,))
                                    }),
    '''

    paramCoarse = {
    'Fisher' : (algs.FisherClass(), {
                                    'Fisher_thresh': np.random.uniform(low=0.2, high=1, size=(ntrial,))
                                    })
    }
    '''
    'L1' : (algs.L1Class(), {
                            'L1_thresh': np.random.uniform(low=1e-5, high=0.1, size=(ntrial,)),
                            'L1_regwgt': np.random.uniform(low=1e-5, high=1, size=(ntrial,))
                            })
    }'''
    accuracies = {}
    for learnername in paramCoarse:
        accuracies[learnername] = np.zeros((ntrial,K,2),dtype = np.float32)

    kf = KFold(n_splits=K)

    gen = kf.split(x)
    if kf_dictSelect:

        #load to check kf contents
        infile = open('kfsplits_5','rb')
        new_dict=pickle.load(infile)
        infile.close()
        gen = gen_kf(new_dict)
    else:
        kf_dict = {}
        for split, (train_index, test_index) in enumerate(kf.split(x)):
            kf_dict[split] = (train_index, test_index)
        outfile = open('kfsplits_5', 'wb')
        pickle.dump(kf_dict, outfile)
        outfile.close()

    splits = {}
    for split, (train_index, test_index) in enumerate(gen):
        splits[split] = (train_index,test_index)

    for k,item in paramCoarse.items():
        print('Selecting values for algorithm %s ..'%k)
        algorithm = item[0]
        params = item[1]
        for i in range(ntrial):
            cur_param = {}
            for p,values in params.items():
                cur_param[p] = values[i]

            for split, (_,(train_index, test_index)) in enumerate(splits.items()):
                algorithm.reset(cur_param)
                print('%s Running with parameters %s ..'%(''*10,algorithm.getparams()))

                trainx, trainy = x[train_index,...], y[train_index]
                testx, testy = x[test_index,...], y[test_index]
                algorithm.select(trainx,trainy)
                algorithm.train(trainx,trainy,testx,testy)
                ypredict = algorithm.predict(testx)
                cur_accuracy = float(sum(ypredict==testy))/testy.shape[0]
                subfeat = algorithm.subfeat.shape[0]
                accuracies[k][i,split][0] = cur_accuracy
                accuracies[k][i,split][1] = subfeat

    print(accuracies)
    outfile = open('kfold_acc_fisher', 'wb')
    pickle.dump((paramCoarse['Fisher'][1], accuracies), outfile)
    outfile.close()

def std_hist(train_x):
    plt.hist(train_x.std(0))
    plt.title('Histogram of number of features in a std bin')
    plt.xlabel("Standard deviation")
    plt.ylabel("N features")
    plt.savefig('hist_1024.png')
    plt.show()
    return

def main():
    train_x,train_y = load_data(os.path.join(args.path,'train_feat.npy'))
    test_x,test_y = load_data(os.path.join(args.path,'test_feat.npy'))
    #TODO read from numpy basaed on dataset
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    global nfeat
    nfeat = test_x.shape[1]
    nclass = np.unique(test_y).shape[0]
    (trainx,trainy),(validx,validy) = split_data(train_x,train_y)

    #TODO hype selection
    hyper_selection(train_x,train_y, 1) #3rd param = save kf_dict
    '''
    #TODO algorithms file --> Done
    classalgs = {'Fisher':algs.FisherClass()
            }
    classalgs['Fisher'].select(train_x,train_y)
    classalgs['Fisher'].train(trainx,trainy,validx,validy)
    prediction = classalgs['Fisher'].predict(test_x)

    #TODO Evaluation metric confusion
    cm = confusion_matrix(test_y,prediction)
    print(cm)
    plot_cm(cm,classes)
    print("label    precision    recall")
    for label in range(nclass):
        print("%5d, %9.3f, %9.3f"%(label,precision(label, cm),recall(label, cm)))

    print("precision total:", precision_macro_average(cm))
    print("recall total:", recall_macro_average(cm))
    print("Accuracy:", ((sum((test_y==prediction))*1.)/test_y.shape[0])*100)
    '''
    #TODO Evaluation metric pairt test
    '''
    dF = K - 1
    for learnername in paramCoarse: #mean for each trial
        for t in range(ntrial):
            avg_acc[learnername][t] = np.mean(accuracies[learnername][t],0)[0]       
        
    tstat, pstat = stats.ttest_rel(avg_acc['Fisher'], avg_acc['MP'])
    tstats, pstat = stats.ttest_rel(avg_acc[BEST], avg_acc['L1'])
    print('Done')
    '''
main()

