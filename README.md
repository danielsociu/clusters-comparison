# Unsupervised clustering

This repository is for the the masters project of comparing two unsupervised clustering models on an image dataset.

It also has two feature extraction approaches.

## How to run:

0. Download the dataset from: https://www.kaggle.com/datasets/andrewmvd/animal-faces
1. Install the requirements `pip install -r requirements.txt`
2. You can use dbscan.sh or just run the python script directly in terminal:
    * `bash dbscan.sh`
    * `python unsupervised_clustering.py` -- this has to be ran with flags:
        *   ```
            usage: unsupervised_clustering.py [-h] [--plot] [--verbose] [--grid-search] [--limit-train] [--standard-scaler] [--type {DBSCAN,AGLOMERATIVE}] [--feature-type {resnet,orb}]
                                  [--mode {unsupervised,random,supervised}]
            optional arguments:
            -h, --help            show this help message and exit
            --plot                Whether to plot stuffs
            --verbose             Whether to print extra info
            --grid-search         Whether to use a grid-search strategy
            --limit-train         wether to limit train to same amount of data as val
            --standard-scaler     whether to scale the orb output
            --type {DBSCAN,AGLOMERATIVE}
                                    Type of model to train
            --feature-type {resnet,orb}
                                    Type of feature engineering
            --mode {unsupervised,random,supervised}
                                    Type of training
            ```