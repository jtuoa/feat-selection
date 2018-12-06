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
    paramCoarse = {
    'Fisher' : (algs.FisherClass(), {
                                    'Fisher_thresh': np.random.uniform(low=0.2, high=1, size=(ntrial,))
                                    })
    'L1' : (algs.L1Class(), {
                            'L1_thresh': np.random.uniform(low=1e-5, high=0.1, size=(ntrial,)),
                            'L1_regwgt': np.random.uniform(low=1e-5, high=1, size=(ntrial,))
                            }),
    'MP' : (algs.MPClass(), {
                            'MP_eps': np.asarray(np.random.uniform(low=1e-5, high=1, size=(ntrial,)))
                            })
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
            outfile = open('ptest_MP', 'wb')
            pickle.dump((paramCoarse['MP'][1],accuracies), outfile)
            outfile.close()


    print(accuracies)
    outfile = open('ptest_MP', 'wb')
    pickle.dump((paramCoarse['MP'][1],accuracies), outfile)
    outfile.close()

def std_hist(train_x):
    plt.hist(train_x.std(0))
    plt.title('Histogram of number of features in a std bin')
    plt.xlabel("Standard deviation")
    plt.ylabel("N features")
    plt.savefig('hist_1024.png')
    plt.show()
    return

def plot_param(kfoldres,algoname):

    params,acc = kfoldres
    ntrial = 10
    fig, ax = plt.subplots()
    acc_mean = np.round(np.mean(acc[algoname],1)[:,0]*100)
    #ax.scatter(range(ntrial),acc_mean)
    feat_mean =  np.mean(acc[algoname],1)[:,1]
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, ntrial))
    #plt.scatter(11, 1, color='white',label = '(regwgt,thresh)')
    #plt.scatter(11, 1, color='white',label = 'Fisher_thresh')
    plt.scatter(11, 1, color='white',label = 'MP_eps')
    for x,(y, c) in enumerate(zip(acc_mean, colors)):
        #lbl = "(%f,%f)"%(params['L1_regwgt'][x], params['L1_thresh'][x])
        #lbl = "%f"%(params['Fisher_thresh'][x])
        lbl = "%f"%(params['MP_eps'][x])
        plt.scatter(x, y, color=c,label = lbl)
        txt = "%d"%feat_mean[x]
        ax.annotate(txt, (x, acc_mean[x]),fontsize='large')
    ax.legend(fontsize='large')
    ax.tick_params(direction='out', labelsize='medium')
    plt.xticks(np.arange(ntrial), range(ntrial))
    #plt.title('Hyperparameters tuning for %s'%algoname,fontsize='large')
    plt.title('Hyperparameters tuning for Matching Pursuit',fontsize='large')
    plt.xlabel('Trial',fontsize='large')
    plt.ylabel('Accuracy',fontsize='large')
    plt.savefig('%s.png'%algoname,bbox_inches='tight')
    #plt.show()

def nselected_test(train_x, train_y, test_x, test_y, alg):
    res = {}
    for i in range(1, 100, 10):
        alg.setoneparam('nselected', i)
        print('Nselected %d'%i)
        alg.select(train_x, train_y)
        alg.train(train_x, train_y, test_x, test_y)
        prediction = alg.predict(test_x)
        acc = ((sum((test_y==prediction))*1.)/test_y.shape[0])*100
        res[i] = acc
        print("Accuracy:", acc)

    return res


def redundancy_test(train_x, train_y, alg):
    dup = np.concatenate((train_x, train_x), axis=-1)
    alg.select(dup, train_y)
    count = 0
    for i in range(1024):
        if (i in alg.subfeat and not i+1024 in alg.subfeat) or (i+1024 in alg.subfeat and not i in alg.subfeat):
            count+=1

    count_ratio = count/1024.0
    return count_ratio

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

    #TODO test algo
    '''
    classalgs = {'L1': algs.L1Class({'L1_thresh': 0.042, 'L1_regwgt': 0.70}),#0.0425712869663748, 0.7046688408267813
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
    '''

    #TODO Evaluation metric pairt test
    '''
    avg_acc_fisher = np.array([0.9966, 0.998, 0.9972, 0.997, 0.997, 0.9966, 0.9966, 0.997, 0.998, 0.997])
    avg_acc_MP = np.array([0.9964,0.997,0.9966,0.9964,0.9964,0.9958,0.9966,0.9962,0.997,0.9958])
    avg_acc_L1 = np.array([0.9972,0.9972,0.9972,0.9956,0.9968,0.9958,0.9978,0.9964,0.997 ,0.9964])
    import matplotlib.mlab as mlab
    mean = avg_acc_fisher.mean()
    sigma = avg_acc_fisher.std()
    x = np.linspace(-3 * sigma + mean, 3 * sigma + mean, 100)
    plt.plot(x,mlab.normpdf(x, mean, sigma),label='Fisher')

    mean = avg_acc_L1.mean()
    sigma = avg_acc_L1.std()
    x = np.linspace(-3 * sigma + mean, 3 * sigma + mean, 100)
    plt.plot(x,mlab.normpdf(x, mean, sigma),label='L1')

    mean = avg_acc_MP.mean()
    sigma = avg_acc_MP.std()
    x = np.linspace(-3 * sigma + mean, 3 * sigma + mean, 100)
    plt.plot(x,mlab.normpdf(x, mean, sigma),label='MP')

    plt.xlabel('Accuracy',fontsize='large')
    plt.yticks([])

    plt.legend(fontsize='large')
    plt.savefig('ptest.png')
    plt.show()
    '''
    '''
    #dF = K - 1
    tstat, pstat = stats.ttest_rel(avg_acc_fisher, avg_acc_MP)
    print(tstat,'  ', pstat)
    tstat, pstat = stats.ttest_rel(avg_acc_L1, avg_acc_MP)
    print(tstat,'  ', pstat)
    res = stats.f_oneway(avg_acc_fisher,avg_acc_L1,avg_acc_MP)
    #tstats, pstat = stats.ttest_rel(avg_acc[BEST], avg_acc_MP)
    print('Done')
    '''

    #TODO redundancy test
    '''
    classalgs = {'L1': algs.L1Class({'L1_thresh': 0., 'L1_regwgt': 0.70, 'nselected': 1024}),#0.0425712869663748, 0.7046688408267813
    'Fisher': algs.FisherClass({'Fisher_thresh': 0., 'nselected':1024}),#0.22109170908304138})
    'MP':algs.MPClass({'MP_eps':0.1})
    }

    count_r = {}
    for k, alg in classalgs.items():
        count_r[k] = redundancy_test(train_x, train_y, alg)

    print("redundancy test: ", count_r)
    '''

    #TODO nselected test
    '''
    final_res = {710: np.array([89.89, 89.96, 89.81]), 10: np.array([86.17, 87.22, 65.07]), 910: np.array([89.9 , 89.87, 89.86]), 210: np.array([89.88, 89.63, 89.66]), 410: np.array([89.93, 89.84, 89.72]), 610: np.array([89.9 , 89.95, 89.82]), 810: np.array([89.89, 89.87, 89.81]), 110: np.array([89.86, 89.23, 89.32]), 1010: np.array([89.91, 89.87, 89.9 ]), 310: np.array([89.95, 89.93, 89.66]), 510: np.array([89.9 , 89.85, 89.83])}
    final_res = {1: np.array([32.22, 32.22, 10.  ]), 71: np.array([89.73, 89.64, 89.24]), 41: np.array([89.34, 89.3, 89.17]), 11: np.array([86.17, 87.22, 65.07]), 81: np.array([89.68, 89.7, 89.38]), 51: np.array([89.55, 89.64, 88.82]), 21: np.array([88.29, 88.74, 83.86]), 91: np.array([89.79, 89.8, 89.47]), 61: np.array([89.63, 89.64, 89.21]), 31: np.array([88.92, 89.28 , 87.24])}
    N = 3
    nrun = len(sorted(final_res))
    ind = np.zeros(nrun)  # the x locations for the groups
    width = 20       # the width of the bars
    #width = 3       # the width of the bars

    fig = plt.figure()#figsize=(5, 6))
    ax = fig.add_subplot(111)

    yvals = np.zeros((N,nrun))
    for r, k in enumerate(sorted(final_res)):
        ind[r] = k
        for i in range(N):
            yvals[i][r] = final_res[k][i]

    rects1 = ax.bar(ind, yvals[0,:], width, color='r')
    rects2 = ax.bar(ind+width, yvals[1,:], width, color='g')
    rects3 = ax.bar(ind+width*2, yvals[2,:], width, color='b')

    ax.tick_params(labelsize='large')
    ax.set_xlabel('N Features Selected',fontsize='large')
    ax.set_ylabel('Accuracy',fontsize='large')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(ind.astype(np.int))#('2011-Jan-4', '2011-Jan-5', '2011-Jan-6') )
    ax.set_xlim(right=1200)
    ax.legend( (rects1[0], rects2[0], rects3[0]), ('Fisher', 'Matching Pursuit', 'L1'),fontsize='large')

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    plt.savefig('nselected.png')
    plt.show()
    '''

    '''
    classalgs = {
    'L1': algs.L1Class({'L1_thresh': 0., 'L1_regwgt': 0.70, 'nselected': 1024}),#0.0425712869663748, 0.7046688408267813
    'Fisher': algs.FisherClass({'Fisher_thresh': 0., 'nselected':1024}),
    'MP':algs.MPClass({'MP_eps':0.0, 'nselected':1024})
    }
    final_res ={}
    for k, alg in classalgs.items():
        print('Nselected alg %s'%k)
        res = nselected_test(train_x, train_y, test_x, test_y, alg)
        for kr, vr in res.items():
            if kr in final_res:
                final_res[kr] = np.append(final_res[kr],[vr])
            else:
                final_res[kr] = np.asarray([vr])
        print(final_res)
    outfile = open('nselected_test_MP_20','wb')
    pickle.dump(final_res,outfile)
    outfile.close()
    '''

    #TODO tsne
    #alg = algs.L1Class({'L1_thresh':0.042, 'L1_regwgt':0.7})
    #alg = algs.FisherClass({'Fisher_thresh': 0.})
    #alg.select(train_x,train_y)
    '''
    selected = np.asarray(range(1024))#alg.subfeat
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    X_2d = tsne.fit_transform(train_x[:,selected])
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(range(nclass), colors, classes):
        plt.scatter(X_2d[train_y == i, 0], X_2d[train_y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()

    plt.savefig('tsne_1024.png')
    #plt.show()
    '''

main()

