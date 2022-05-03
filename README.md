This repository contains code and models for our NAACL 2022 paper **What kinds of errors do reference resolution models make and what can we learn from them?** by Jorge Sánchez, Mauricio Mazuecos, Hernán Maina and Luciana Benotti.


# Installation


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

# Setup the training data


Clone the [Referring Expression Dataset API](https://github.com/lichengunc/refer)

```sh
$ git clone https://github.com/lichengunc/refer.git && cd refer
$ git checkout python3
```

and follow the instructions to access the ReferItGame (a.k.a RefCLEF), RefCOCO, RefCOCO+ and RefCOCOg datasets.


# Training and validation


Run

```sh
$ python3 trainval.py -h
```

for a complete list of training options.


# Pretrained models


[Here](https://drive.google.com/drive/folders/1ud7RaR_0rmJws4xGJeGz-tdZMugvd2eh?usp=sharing) you can find both the baseline and extended models trained on the different datasets (Table 3 in the paper). For convenience, we recommend to keep the same directory structure since the testing script infer some of the parameters from the path names.

* ReferItGame: [baseline](https://drive.google.com/drive/folders/1Yd0wVAGne5-drWz8wwlPjkIH6pZItzqm?usp=sharing), [extended](https://drive.google.com/drive/folders/1aPNzpfpeb0Y7Ztba-7N4EiR03LRqWzGg?usp=sharing)
* RefCOCO: [baseline](https://drive.google.com/drive/folders/1Zm92kg3ereWMSUqlqJocd9tG5dcI0U4y?usp=sharing), [extended](https://drive.google.com/drive/folders/1xTDmJzxJ_KbrmKj6DkBLqNyZtdkbcD6z?usp=sharing)
* RefCOCO+: [baseline](https://drive.google.com/drive/folders/1KxYomKbBTBEAWeB7DrnixwBavc44KZ3p?usp=sharing), [extended]()
* RefCOCOg: [baseline](https://drive.google.com/drive/folders/1YXw1Nt0gy34aaemOZJpigGvMq72Of2Zy?usp=sharing), [extended]()


# Evaluation


First, you'll a running version of stanza. You can download the english package files as:

```sh
$ python3 -c "import stanza; stanza.download('en')"
```

You can also use spacy, in which case you need to change the ```backend="stanza"``` argument in line 178 to "backend=spacy". To get the spacy language files, run:

```sh
$ python3 -m spacy download en_core_web_md
```

Now, to test a trained model run:

```sh
$ python3 test.py <MODEL.ckpt>
```

The script will inferr the dataset and parameters from the file path. You can run ```-h``` and check which options are available. The test script is provided as an example use of our trained models. You can customize it to your needs.

# Error analysis annotation

We make available the annotation of the type of abilities needed for each RE to
be correctly resolved in the file
```ReferIt_Skill_annotation-NAACL2022.csv.g0```.

The file contains more type of abilities than the ones discussed in the paper.
The only types relevant for the analysis are:

 - fuzzy objects
 - meronimy
 - occlusion
 - directional
 - implicit
 - typo
 - viewpoint
