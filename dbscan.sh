python3 unsupervised_clustering.py \
    --verbose \
    --mode 'random' \
    --feature-type 'resnet' \
    # --feature-type 'orb' \
    # --grid-search
    # --type 'DBSCAN' \
    # --type 'AGLOMERATIVE' \
    # --standard-scaler
    # --limit-train \

# usage: unsupervised_clustering.py [-h] [--plot] [--verbose] [--grid-search] [--limit-train] [--standard-scaler] [--type {DBSCAN,AGLOMERATIVE}] [--feature-type {resnet,orb}]
#                                   [--mode {unsupervised,random,supervised}]

# optional arguments:
#   -h, --help            show this help message and exit
#   --plot                Whether to plot stuffs
#   --verbose             Whether to print extra info
#   --grid-search         Whether to use a grid-search strategy
#   --limit-train         wether to limit train to same amount of data as val
#   --standard-scaler     whether to scale the orb output
#   --type {DBSCAN,AGLOMERATIVE}
#                         Type of model to train
#   --feature-type {resnet,orb}
#                         Type of feature engineering
#   --mode {unsupervised,random,supervised}
#                         Type of training