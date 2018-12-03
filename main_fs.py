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
from sklearn.manifold import TSNE
n_sne = 50000

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
    ntrial = 1 #10
    K = 10
    '''
    'MP' : (algs.MPClass(), {
                            'MP_eps': np.random.uniform(low=1e-5, high=1, size=(ntrial,))
                            }),
    'Fisher' : (algs.FisherClass(), {
                                    'Fisher_thresh': np.random.uniform(low=1e-5, high=2e5, size=(ntrial,))
                                    }),

    paramCoarse = {
    'Fisher' : (algs.FisherClass(), {
                                    'Fisher_thresh': np.random.uniform(low=0.2, high=1, size=(ntrial,))
                                    })
    }
    
    'L1' : (algs.L1Class(), {
                            'L1_thresh': np.random.uniform(low=1e-5, high=0.1, size=(ntrial,)),
                            'L1_regwgt': np.random.uniform(low=1e-5, high=1, size=(ntrial,))
                            })
    }'''
    
    paramCoarse = {
    'Fisher' : (algs.FisherClass(), {'Fisher_thresh': [0.22109170908304138]})
    }
    
    accuracies = {}
    for learnername in paramCoarse:
        accuracies[learnername] = np.zeros((ntrial,K,2),dtype = np.float32)

    kf = KFold(n_splits=K)

    gen = kf.split(x)
    if kf_dictSelect:

        #load to check kf contents
        infile = open('kfsplits_%d'%K,'rb')
        new_dict=pickle.load(infile)
        infile.close()
        gen = gen_kf(new_dict)
    else:
        kf_dict = {}
        for split, (train_index, test_index) in enumerate(kf.split(x)):
            kf_dict[split] = (train_index, test_index)
        outfile = open('kfsplits_%d'%K, 'wb')
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
    outfile = open('ptest_fisher', 'wb')
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

def redundancy_test(train_x, train_y, alg):    
    dup = np.concatenate((train_x, train_x), axis=-1)
    alg.select(dup, train_y)
    count = 0
    for i in range(1024):
        if (i in alg.subfeat and not i+1024 in alg.subfeat) or (i+1024 in alg.subfeat and not i in alg.subfeat):
            count+=1
        
    count_ratio = count/1024.0
    #count1 = np.sum(alg.subfeat < 1024)/1024.
    #count2 = 1. - count1
    #count = max(count1, count2)
    return count_ratio
    
def nselected_test(train_x, train_y, test_x, test_y, alg):
    res = {}
    for i in range(10, 1024, 100):
        alg.setoneparam('nselected', i)
        alg.select(train_x, train_y)
        alg.train(train_x, train_y, test_x, test_y)
        prediction = alg.predict(test_x)
        acc = ((sum((test_y==prediction))*1.)/test_y.shape[0])*100
        res[i] = acc
        print("Accuracy:", acc)
        
    return res


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
    #hyper_selection(train_x,train_y, 1) #1= dont gen kf_split
    
    #TODO algorithms file --> Done
    '''
    classalgs = {'L1': algs.L1Class({'L1_thresh': 0.042, 'L1_regwgt': 0.70})#0.0425712869663748, 0.7046688408267813
    #'Fisher': algs.FisherClass({'Fisher_thresh': 0.22})#0.22109170908304138})
    }

    classalgs['L1'].select(train_x,train_y)
    print('Number of features selected ',classalgs['L1'].subfeat.shape[0])
    classalgs['L1'].train(train_x,train_y,test_x,test_y)
    prediction = classalgs['L1'].predict(test_x)
	
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
    
    #TODO variance test
    '''
    avg_run = {}
    outfile = pickle.load(open('kfold_acc_l1_10t_5f', 'rb'))
    avg_run =np.mean(outfile[1]['L1'],1)[:,0]
    avg_var = np.var(avg_run)
    pdb.set_trace()
    '''
    
    #TODO Evaluation metric pairt test
    '''
    avg_acc_fisher = np.array([0.9966, 0.998, 0.9972, 0.997, 0.997, 0.9966, 0.9966, 0.997, 0.998, 0.997])
    #avg_acc_MP = np.array([])
    #avg_acc_L1 = np.array([0.9972,0.9972,0.9972,0.9956,0.9968,0.9958,0.9978,0.9964,0.997 ,0.9964])
    bla = np.array([0.6,0.8,0.8,0.4,0.5,0.3,0.4,0.64,0.7 ,0.4])
    #dF = K - 1   
    tstat, pstat = stats.ttest_rel(avg_acc_fisher, bla)
    pdb.set_trace()
    #tstats, pstat = stats.ttest_rel(avg_acc[BEST], avg_acc_MP)
    print('Done')
    '''
    
    #TODO redundancy test
    '''
    classalgs = {'L1': algs.L1Class({'L1_thresh': 0., 'L1_regwgt': 0.70, 'nselected': 1024}),#0.0425712869663748, 0.7046688408267813
    'Fisher': algs.FisherClass({'Fisher_thresh': 0., 'nselected':1024})#0.22109170908304138})
    }
    count_r = {}
    for k, alg in classalgs.items():
        count_r[k] = redundancy_test(train_x, train_y, alg)
        
    print("redundancy test: ", count_r)
    '''
    '''
    #TODO nselected test
    classalgs = {
    'L1': algs.L1Class({'L1_thresh': 0., 'L1_regwgt': 0.70, 'nselected': 1024}),#0.0425712869663748, 0.7046688408267813
    'Fisher': algs.FisherClass({'Fisher_thresh': 0., 'nselected':1024})#0.22109170908304138})
    }
    final_res ={}
    for k, alg in classalgs.items():
        res = nselected_test(train_x, train_y, test_x, test_y, alg)
        for kr, vr in res.items():
            if kr in final_res:
                final_res[kr] = np.append(final_res[kr],[vr])
            else:
                final_res[kr] = np.asarray([vr])
        print(final_res)
    '''
    #TODO tsne 
    #alg = algs.L1Class({'L1_thresh':0.042, 'L1_regwgt':0.7})
    alg = algs.FisherClass({'Fisher_thresh': 0.})
    alg.select(train_x,train_y)
    selected = alg.subfeat
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    X_2d = tsne.fit_transform(train_x[:,selected])
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(range(nclass), colors, classes):
        plt.scatter(X_2d[train_y == i, 0], X_2d[train_y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()

           
main()

