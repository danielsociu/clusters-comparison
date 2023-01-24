import plotly
from sklearn.decomposition import PCA
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

from PIL import Image
import torch
from torchvision.models import resnet18
from torchvision import transforms

from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score


def get_classes(data_path):
    class_to_idx = {}
    for index, folder_name in enumerate(os.listdir(data_path)):
        class_to_idx[folder_name] = index
    return class_to_idx


def liniarize_images(dataset):
    for data in dataset:
        data['image'] = data['image'].flatten()
    return dataset


def extract_images(dataset):
    return [data['image'] for data in dataset]


def read_featured_data(data_path, model, transformer, device):
    all_data = []
    model.eval()
    for class_name in CLASS_NAMES.keys():
        folder_path = data_path / class_name
        for index, img_name in tqdm(enumerate(os.listdir(folder_path))):
            full_image_path = folder_path / img_name
            full_image = Image.open(full_image_path)
            processed_image = transformer(full_image).unsqueeze(0).to(device)
            features = model(processed_image).flatten()
            all_data.append({
                "label": CLASS_NAMES[class_name],
                "image": features.detach().cpu().numpy(),
                "path": str(full_image_path)
            })
    return all_data


def class_matcher(search_data, predictions):
    results = {}
    all_classes_counters = {}
    for class_name in CLASS_NAMES.keys():
        all_classes_counters[class_name] = dict(class_correct_counter(
            CLASS_NAMES[class_name], search_data, predictions))
    all_classes_counters_copy = deepcopy(all_classes_counters)
    while len(all_classes_counters) > 0:
        pair_found = get_pair_key_of_max(all_classes_counters)
        results[pair_found[0]] = pair_found[1]

    return results, all_classes_counters_copy


def get_pair_key_of_max(cls_counters):
    max_found = -1
    for class_name in cls_counters:
        for pred, freq in cls_counters[class_name].items():
            if max_found < freq:
                max_found = freq
                pair_keys = (class_name, pred)
    # delete the max found
    del cls_counters[pair_keys[0]]

    classes_to_delete = []
    for class_name in cls_counters:
        if pair_keys[1] in cls_counters[class_name]:
            del cls_counters[class_name][pair_keys[1]]
            if len(cls_counters[class_name]) == 0:
                classes_to_delete.append(class_name)
    for key in classes_to_delete:
        del cls_counters[key]
    return pair_keys


def class_correct_counter(class_verified, dataset, predictions):
    classes_counter = defaultdict(lambda: 0)
    for index in range(len(dataset)):
        if predictions[index] != -1 and class_verified == dataset[index]['label']:
            classes_counter[predictions[index]] += 1
    return classes_counter


def filter_predictions(predictions, actual_classes, class_to_id):
    filtered_preds = []
    pred_to_label = get_pred_to_label_dict(actual_classes, class_to_id)
    print(pred_to_label)
    for pred in predictions:
        if pred in class_to_id.values():
            filtered_preds.append(pred_to_label[pred])
        else:
            filtered_preds.append(-1)
    return filtered_preds


def get_pred_to_label_dict(actual_classes, class_to_id):
    pred_to_label = {}
    for class_name in actual_classes:
        if class_name in class_to_id.keys():
            pred_to_label[class_to_id[class_name]] = actual_classes[class_name]
    return pred_to_label


def calculate_accuracy(dataset, predictions, actual_classes, class_to_id):
    labels = [data['label'] for data in dataset]
    final_preds = filter_predictions(predictions, actual_classes, class_to_id)
    return accuracy_score(labels, final_preds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='Whether to plot stuffs')
    parser.add_argument('--verbose', action='store_true', help='Whether to print extra info')
    parser.add_argument('--type', type=str, default=DBSCAN, choices=['DBSCAN'], help='Type of model to train')
    args = parser.parse_args()
    return args


TRAIN_DATA_PATH = Path('./afhq/train')
VAL_DATA_PATH = Path('./afhq/val')
CLASS_NAMES = get_classes(TRAIN_DATA_PATH)


def main_dbscan_resnet(args):
    # class specific initializations:
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
    
    # read the data
    train_data = read_featured_data(
        TRAIN_DATA_PATH, RESNET_EXTRACTOR, IMAGE_EXTRACTOR, DEVICE)
    val_data = read_featured_data(
        VAL_DATA_PATH, RESNET_EXTRACTOR, IMAGE_EXTRACTOR, DEVICE)
    
    # delete the resnet model
    del RESNET_EXTRACTOR
    torch.cuda.empty_cache()

    # extract images
    X = extract_images(train_data)
    X_val = extract_images(val_data)


    if args.plot:
        pca = PCA(n_components=512)
        points = pca.fit_transform(X)
        points_val = pca.fit_transform(X_val)
        fig = plotly.subplots.make_subplots(rows=2, cols=1)
        fig.add_trace(
            plotly.express.scatter(points, x=0, y=1),
            row=1, col=1
        )
        fig.add_trace(
            plotly.express.scatter(points_val, x=0, y=1),
            row=1, col=1
        )
        fig.update_layout(height=800, width=800, title_text="Train points vs val points")
        fig.show()

    # predfict
    clustering = DBSCAN(eps=14, min_samples=25).fit(X)
    train_preds = clustering.labels_
    val_preds = clustering.fit_predict(X_val)

    # get class matchers given predictions and labels
    class_to_preds, all_classes = class_matcher(train_data, train_preds)

    if args.verbose:
        print("All classes:")
        print(all_classes)
        print("Matched classes:")
        print(class_to_preds)

        print("Predictions stats on train:")
        print(f'Different than -1: {np.count_nonzero(clustering.labels_ == -1)}')
        print(f'Max predicted class: {max(clustering.labels_)}')
        print(f'Length of x: {len(X)}')

    print(f'Final accuracy on train dataset:    {calculate_accuracy(train_data, train_preds, CLASS_NAMES, class_to_preds)}')
    print(f'Final accuracy on val dataset:      {calculate_accuracy(val_data, val_preds, CLASS_NAMES, class_to_preds)}')


if __name__ == "__main__":
    args = parse_args()
    if args.type == "DBSCAN":
        main_dbscan_resnet(args)
