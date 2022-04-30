This repository contains code and models for our NAACL 2022 paper **What kinds of errors do reference resolution models make and what can we learn from them?** by Jorge Sánchez, Mauricio Mazuecos, Hernán Maina and Luciana Benotti.

1. Installation
===============

Setup the code in a virtualenv

```sh
$ git clone https://github.com/jadrs/rec.git && cd rec
$ python3 -m venv venv  && source venv/bin/activate
```

you'll also need a running version of [pytorch](https://pytorch.org/get-started/locally/). You can go to the website and choose the version that best suits your hardware, eg.:

```sh
$ python3 -m pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
$ python3 -m pip install -r requirements.txt
```

2. Setup the training data
==========================

Clone the [Referring Expression Dataset API](https://github.com/lichengunc/refer)

```sh
$ git clone https://github.com/lichengunc/refer.git
$ git checkout python3
```

and follow the instructions to access the ReferItGame (a.k.a RefCLEF), RefCOCO, RefCOCO+ and RefCOCOg datasets.


3. Training and validation
==========================

Run

```sh
$ python3 trainval.py -h
```

for a complete list of training options.

4. Pretrained models
====================

[Here](https://drive.google.com/drive/folders/1ud7RaR_0rmJws4xGJeGz-tdZMugvd2eh?usp=sharing) you can find both the baseline and extended models trained on the different datasets (Table 3 in the paper). For convenience, we recommend to keep the same directory structure since the testing script infer some of the parameters from the path names.

* ReferItGame: [baseline](), [extended]()
* RefCOCO: [baseline](), [extended]()
* RefCOCO+: [baseline](), [extended]()
* RefCOCOg: [baseline](), [extended]()

5. Testing
==========

For testing a trained model run:

```sh
$ python3 test.py <MODEL.ckpt>
```

The
