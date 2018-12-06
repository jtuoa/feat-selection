# feature-selection
Mini-project CMPUT 566


## Installation
```bash
virtualenv -p python3 .env
source .env/bin/activate
```

* Install required packages 
```bash
pip install -r req.txt 
```

## VGG
```bash
python cifarvgg.py --mode train --dataset cifar10 --out_dir cifar10vgg #train 
python cifarvgg.py --mode test --dataset cifar10 --chkpnt cifar10vgg/model.hdf5 #test
python cifarvgg.py --mode extract --dataset cifar10 --chkpnt cifar10vgg/model.hdf5 #extract features
```

## Feature selection
```bash
python main_fs.py --path cifar10vgg
```
