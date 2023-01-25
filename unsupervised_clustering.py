from sklearn.decomposition import PCA
import random
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import argparse
import plotly.express as px

from PIL import Image
import torch
from torchvision.models import resnet18
from torchvision import transforms

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

SEED = 42
random.seed(SEED)


def get_classes(data_path):
    """
    Gets the classes names and ids given the folder towards train
    Dataset format: train/class_name/*.png 
    """
    class_to_idx = {}
    for index, folder_name in enumerate(os.listdir(data_path)):
        class_to_idx[folder_name] = index
    return class_to_idx


def liniarize_images(dataset):
    """ 
    Apply flatten to images from dataset
    """
    for data in dataset:
        data['image'] = data['image'].flatten()
    return dataset


def extract_images(dataset):
    """Return the array of images"""
    return [data['image'] for data in dataset]


def read_featured_data(data_path, model, transformer, device):
    """
    Read the data using the resnet approach. (with the model)
    """
    all_data = []
    model.eval()

    for class_name in CLASS_NAMES.keys():
        folder_path = data_path / class_name

        # Iterate through images
        for index, img_name in tqdm(enumerate(os.listdir(folder_path))):
            full_image_path = folder_path / img_name
            full_image = Image.open(full_image_path)

            # apply the transforms correspondent to resnet model
            processed_image = transformer(full_image).unsqueeze(0).to(device)
            # apply the model and flatten the output
            features = model(processed_image).flatten()

            # add all data
            all_data.append({
                "label": CLASS_NAMES[class_name],
                "image": features.detach().cpu().numpy(),
                "path": str(full_image_path)
            })
    return all_data


def read_featured_data_orb(data_path, orb):
    """
    Read the data using the ORB approach. (with the model)
    """
    all_data = []
    # Create the backup orb in case the original doesn't generate any keypoints/descriptors
    backup_orb = cv2.ORB_create(
        nfeatures=orb.getMaxFeatures(), fastThreshold=0, edgeThreshold=0)

    for class_name in CLASS_NAMES.keys():
        folder_path = data_path / class_name

        # iterate through images
        for index, img_name in tqdm(enumerate(os.listdir(folder_path))):
            full_image_path = folder_path / img_name
            full_image = cv2.imread(str(full_image_path), cv2.IMREAD_COLOR)

            # Get the features of the keypoints
            _, features = orb.detectAndCompute(full_image, None)
            if features is None:
                _, features = backup_orb.detectAndCompute(full_image, None)

            # flatten the features and then pad them to have size 3200 and scale to 0-1
            features = features.flatten()
            features = np.pad(features, (0, orb.getMaxFeatures(
            )*32 - features.shape[0]), 'constant', constant_values=(0, features[-1]))
            features = features / 255.

            all_data.append({
                "label": CLASS_NAMES[class_name],
                "image": features,
                "path": str(full_image_path)
            })
    return all_data


def read_labels(data_path):
    """
    Read only the labels from the folder (no images)
    """
    all_data = []

    for class_name in CLASS_NAMES.keys():
        folder_path = data_path / class_name

        # Iterate through images and append them to the all_data
        for index, img_name in tqdm(enumerate(os.listdir(folder_path))):
            full_image_path = folder_path / img_name
            all_data.append({
                "label": CLASS_NAMES[class_name],
                "path": str(full_image_path)
            })
    return all_data


def class_matcher(search_data, predictions, class_names):
    """
    Given the predictions and classes, match the cluster labels with the actual labels 
    Returns: {cluster_class: real_class, ..} an dict of max length: the number of classes
    """
    results = {}
    all_classes_counters = {}

    # iterate through classes
    for class_name in class_names.keys():
        # get the prediction labels corresponding to each class
        all_classes_counters[class_name] = dict(class_correct_counter(
            class_names[class_name], search_data, predictions))

    all_classes_counters_copy = deepcopy(all_classes_counters)

    # get the instance with most matches and make the match between cluster_label and actual_label
    while len(all_classes_counters) > 0:
        # checking any keys to delete
        delete_check = []
        for key in all_classes_counters.keys():
            if len(all_classes_counters[key]) == 0:
                delete_check.append(key)
        for key in delete_check:
            del all_classes_counters[key]
        if len(all_classes_counters) == 0:
            continue
        # get the best cluster_label and actual_label pair
        pair_found = get_pair_key_of_max(all_classes_counters)
        # add it to results
        results[pair_found[0]] = pair_found[1]

    return results, all_classes_counters_copy


def get_pair_key_of_max(cls_counters):
    """
    Finds the best pair of cluster_label and actual_label based on the max frequency
    """
    max_found = -1
    # find the max pair
    for class_name in cls_counters:
        for pred, freq in cls_counters[class_name].items():
            if max_found < freq:
                max_found = freq
                pair_keys = (class_name, pred)

    # delete the max found
    del cls_counters[pair_keys[0]]

    # delete the cluster_label from the other classes
    classes_to_delete = []
    for class_name in cls_counters:
        if pair_keys[1] in cls_counters[class_name]:
            del cls_counters[class_name][pair_keys[1]]
            if len(cls_counters[class_name]) == 0:
                classes_to_delete.append(class_name)
    # delete any class array in case of empty (after deletion of its cluster_label)
    for key in classes_to_delete:
        del cls_counters[key]

    # return the pair found
    return pair_keys


def class_correct_counter(class_verified, dataset, predictions):
    """
    Given a class to be verified, get all the predictions labels counters corresponding to this class
    """
    classes_counter = defaultdict(lambda: 0)
    for index in range(len(dataset)):
        # if there is a predictions and the class verified is matching the current item checked
        if predictions[index] != -1 and class_verified == dataset[index]['label']:
            # add to the frequency counter
            classes_counter[predictions[index]] += 1
    return classes_counter


def filter_predictions(predictions, actual_classes, class_to_id):
    """
    Filters the predictions for the calculation of score.
    Uses the class_matcher to filter the labels
    """
    filtered_preds = []
    # gets the cluster_label to actual_label dictionary
    pred_to_label = get_pred_to_label_dict(actual_classes, class_to_id)
    print(pred_to_label)

    # iterates through predictions and filters them
    for pred in predictions:
        if pred in class_to_id.values():
            filtered_preds.append(pred_to_label[pred])
        else:
            filtered_preds.append(-1)

    return filtered_preds


def get_pred_to_label_dict(actual_classes, class_to_id):
    """
    Given the class_matcher, it creates the cluster_label to actual_label dictionary
    """
    pred_to_label = {}
    for class_name in actual_classes:
        # just removes the string 'class_name' makes the mapping between the values
        if class_name in class_to_id.keys():
            pred_to_label[class_to_id[class_name]] = actual_classes[class_name]
    return pred_to_label


def calculate_accuracy(dataset, predictions, actual_classes, class_to_id):
    """
    Given the dataset, predictions, and classes, calculates the accuracy score
    """
    # get the labels
    labels = [data['label'] for data in dataset]
    # filter the predictions
    final_preds = filter_predictions(predictions, actual_classes, class_to_id)
    # calculate and return the score
    return accuracy_score(labels, final_preds)


def predict(args, model, train, val, class_names):
    """
    Trains and predicts any kind of clustered model
    """
    print('\n Starting test with:')
    print(model)

    # get the data for train
    X = extract_images(train)
    X_val = extract_images(val)
    clustering = model.fit(X)
    train_preds = clustering.labels_
    val_preds = clustering.fit_predict(X_val)

    # get class matchers given predictions and labels
    class_to_preds, all_classes = class_matcher(
        train, train_preds, class_names)

    if args.verbose:
        print("All classes:")
        print(all_classes)
        print("Matched classes:")
        print(class_to_preds)

        print("Predictions stats on train:")
        print(
            f'Different than -1: {np.count_nonzero(clustering.labels_ == -1)}')
        print(f'Max predicted class: {max(clustering.labels_)}')
        print(f'Length of x: {len(X)}')

    # calculate the accuracies for train and val
    train_acc = calculate_accuracy(
        train, train_preds, class_names, class_to_preds)
    val_acc = calculate_accuracy(val, val_preds, class_names, class_to_preds)

    print(f'Final accuracy on train dataset:    {train_acc}')
    print(f'Final accuracy on val dataset:      {val_acc}')
    return train_acc, val_acc


def get_minimum_distance_dbscan(train_data, percent=0.7, max_eps=30):
    """
    Gets the minimum eps from which to start the DBSCAN grid search
    """

    X = extract_images(train_data)
    possible_values = list(range(0, max_eps))
    possible_values[0] = 0.5

    # iterates through the eps values
    for eps in possible_values:
        # trains the model
        model = DBSCAN(eps=eps, min_samples=2).fit(X)
        # checks if number of predicted labels is less than a specific percentage (70% default)
        if np.count_nonzero(model.labels_ == -1) < int(len(X) * percent):
            return eps
    return max_eps


def get_minimum_distance_aglomerative(train_data, max_dist=250):
    """
    Gets the minimum distance_threshold from which to start the DBSCAN grid search
    """

    X = extract_images(train_data)
    possible_values = list(range(10, max_dist, 5))

    # iterates through early dist values
    for dist in possible_values:
        # trains the model
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=dist, compute_full_tree=True).fit(X)
        # check if there are less than 10 different classes
        if len(np.unique(model.labels_)) < 10:
            return dist
    return max_dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot stuffs')
    parser.add_argument('--verbose', action='store_true',
                        help='Whether to print extra info')
    parser.add_argument('--grid-search', action='store_true',
                        help='Whether to use a grid-search strategy')
    parser.add_argument('--limit-train', action='store_true',
                        help='wether to limit train to same amount of data as val')
    parser.add_argument('--standard-scaler', action='store_true',
                        help='whether to scale the orb output')
    parser.add_argument('--type', type=str, default='DBSCAN',
                        choices=['DBSCAN', 'AGLOMERATIVE'], help='Type of model to train')
    parser.add_argument('--feature-type', type=str, default='resnet',
                        choices=['resnet', 'orb'], help='Type of feature engineering')
    parser.add_argument('--mode', type=str, default='unsupervised',
                        choices=['unsupervised', 'random', 'supervised'], help='Type of training')
    args = parser.parse_args()
    return args


TRAIN_DATA_PATH = Path('./afhq/train')
VAL_DATA_PATH = Path('./afhq/val')
CLASS_NAMES = get_classes(TRAIN_DATA_PATH)


def prepare_data_resnet(args):
    """
    Prepare the data for the resnet type feature extractor
    """

    # define RESNET specific variables
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
    base_model = resnet18(pretrained=True)
    RESNET_EXTRACTOR = torch.nn.Sequential(
        *list(base_model.children())[:-1]).to(DEVICE)

    IMAGE_EXTRACTOR = transforms.Compose([
        transforms.Resize(
            256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # converts to [0, 1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # read the data with model feature extraction
    train_data = read_featured_data(
        TRAIN_DATA_PATH, RESNET_EXTRACTOR, IMAGE_EXTRACTOR, DEVICE)
    val_data = read_featured_data(
        VAL_DATA_PATH, RESNET_EXTRACTOR, IMAGE_EXTRACTOR, DEVICE)

    # whether to limit the data to the length of the validation set
    if args.limit_train:
        train_data = random.sample(train_data, len(val_data))

    # delete the resnet model
    del RESNET_EXTRACTOR
    torch.cuda.empty_cache()

    return train_data, val_data


def prepare_data_orb(args):
    """
    Prepare the data for the ORB type feature extractor
    """

    # initialize the orb feature extractor
    orb = cv2.ORB_create(nfeatures=100)
    # read the data given the orb feature descriptor
    train_data = read_featured_data_orb(TRAIN_DATA_PATH, orb)
    val_data = read_featured_data_orb(VAL_DATA_PATH, orb)

    # orb works only in 1500 data samples, unless it's unsupervised training
    if args.mode != "supervised":
        train_data = random.sample(train_data, len(val_data))

    # apply standard scaler if needed (worse results)
    if args.standard_scaler:
        X = extract_images(train_data)
        X_val = extract_images(train_data)
        scaler = StandardScaler()
        scaler.fit(X)

        X = scaler.transform(X)
        X_val = scaler.transform(X_val)

        for index, sample in enumerate(train_data):
            sample['image'] = X[index]
        for index, sample in enumerate(val_data):
            sample['image'] = X_val[index]

    return train_data, val_data


def main_dbscan(args, train_data, val_data):
    """
    Main function that runs the DBSCAN algorithm given any data in the proper format
    """
    # Some plots that were commented since they didn't work in python .py (env problems)
    # those plots workekd in the ipynb
    # extract images
    # X = extract_images(train_data)
    # X_val = extract_images(val_data)

    # if args.plot:
    #     pca = PCA(n_components=2)
    #     points = pca.fit_transform(X)
    #     points_val = pca.transform(X_val)
    #     fig = px.scatter(points, x=0, y=1)
    #     fig.show()
    #     fig = px.scatter(points_val, x=0, y=1)
    #     fig.show()

    # apply the grid search
    if args.grid_search:
        # obtain th minimum eps value from which to start the search from
        min_eps_value = get_minimum_distance_dbscan(train_data)

        # generate the grid values to iterate
        sample_range = list(range(5, 15))
        eps_values = np.linspace(min_eps_value, min_eps_value + 10, 20)

        # just preset bigger range in case of limit training (since it will be faster)
        if args.limit_train:
            min_eps_value = 0.5
            sample_range = list(range(1, 30))
            eps_values = np.linspace(min_eps_value, min_eps_value + 30, 60)

        print(f'The starting value for eps is: {min_eps_value}')
        best_acc = 0
        best_params = {}

        # apply the grid search
        for eps in eps_values:
            for samples in sample_range:
                # define the model
                model = DBSCAN(eps=eps, min_samples=samples)
                # train and predict
                train_acc, val_acc = predict(
                    args, model, train_data, val_data, CLASS_NAMES)
                # get the best accuracy and params
                if best_acc < val_acc:
                    best_acc = val_acc
                    best_params = {
                        'eps': eps,
                        'min_samples': samples
                    }

        # print the best model
        print(f'Best accuracy found {best_acc} with params:')
        print(best_params)
    else:
        # if not gridsearch, just apply the best model found in the grid search
        model = DBSCAN(eps=16.3, min_samples=5)
        if args.feature_type == "orb":
            model = DBSCAN(eps=22.5, min_samples=5)
        if args.limit_train:
            model = DBSCAN(eps=13.12, min_samples=24)

        # train and predict the model
        train_acc, val_acc = predict(
            args, model, train_data, val_data, CLASS_NAMES)


def main_aglomerative(args, train_data, val_data):
    """
    Main function that runs the aglomerative algorithm given any data in the proper format
    """
    # apply the grid search
    if args.grid_search:
        # obtain the min distance from which to start the search
        min_dist_value = get_minimum_distance_aglomerative(train_data)
        dist_values = np.linspace(min_dist_value, min_dist_value + 200, 50)

        # use different if limit_train
        if args.limit_train:
            min_dist_value = 1
            dist_values = np.linspace(
                min_dist_value, min_dist_value + 300, 150)

        print(f'The starting value for dist is: {min_dist_value}')
        best_acc = 0
        best_params = {}

        # apply the grid
        for dist in dist_values:
            # define the model
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=dist, compute_full_tree=True)
            # train the model and predict
            train_acc, val_acc = predict(
                args, model, train_data, val_data, CLASS_NAMES)
            if best_acc < val_acc:
                best_acc = val_acc
                best_params = {
                    'dist': dist
                }

        # print best model
        print(f'Best accuracy found {best_acc} with params:')
        print(best_params)
    else:
        # declare best model for agglomerative
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=343.87, compute_full_tree=True)
        if args.feature_type == "orb":
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=60.4, compute_full_tree=True)
        if args.limit_train:
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=176.16, compute_full_tree=True)

        # train and predict best model
        train_acc, val_acc = predict(
            args, model, train_data, val_data, CLASS_NAMES)


def main_random(args):
    """
    The main for random approach
    """

    # read data here
    val_data = read_labels(VAL_DATA_PATH)
    # make random predictions with random.randing
    random_predictions = [random.randint(
        0, len(CLASS_NAMES) - 1) for _ in range(len(val_data))]

    actual_labels = [sample['label'] for sample in val_data]

    # evaluate accuracy score of the random approach
    random_accuracy = accuracy_score(actual_labels, random_predictions)

    print(f'The score using random preds is: {random_accuracy}')


def main_supervised(args, train_data, val_data):
    """
    The main for the supervised approach, gets the data in the proper format of feature vectors.
    """
    # define the model
    supervised_classifier = svm.SVC()

    # prepare the data
    X = extract_images(train_data)
    X_val = extract_images(val_data)
    y = [sample['label'] for sample in train_data]
    y_val = [sample['label'] for sample in val_data]

    print("Training the SVM classifier")
    supervised_classifier.fit(X, y)

    print("Predicting train data")
    train_preds = supervised_classifier.predict(X)
    print("Predicting val data")
    val_preds = supervised_classifier.predict(X_val)

    # get the accuracy score for the SVM model
    train_acc = accuracy_score(y, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    print(f'Final accuracy on train dataset:    {train_acc}')
    print(f'Final accuracy on val dataset:      {val_acc}')


if __name__ == "__main__":
    # main function that calls one of the previous mains given the line arguments
    args = parse_args()
    if args.feature_type == "orb" and args.limit_train:
        print("CANNOT RUN ORB WITH LIMIT TRAIN")
        exit(-1)
    if args.mode == 'random':
        main_random(args)
    else:
        if args.feature_type == "resnet":
            train_data, val_data = prepare_data_resnet(args)
        elif args.feature_type == "orb":
            train_data, val_data = prepare_data_orb(args)

        if args.mode == "supervised":
            main_supervised(args, train_data, val_data)
        else:
            if args.type == "DBSCAN":
                main_dbscan(args, train_data, val_data)
            elif args.type == "AGLOMERATIVE":
                main_aglomerative(args, train_data, val_data)
