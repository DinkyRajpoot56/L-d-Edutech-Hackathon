from selectors import EpollSelector
import torch
import numpy as np
import cv2
import pandas as pd
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from model import build_model
from dataset import ImageDataset
# Constants and other configurations.
BATCH_SIZE = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 256
NUM_WORKERS = 4
CLASS_NAMES = ['Crack Detected', 'Crack Undetected']
To create the test data loaders we will use a batch size of 1. We use the same image size as in the training, that is, 256×256. We have a CLASS_NAMES list to map the results to the class names. Index 0 will represent that a crack is detected, while index 1 will represent that a crack is not detected.

Next, we have three functions.

test.py
def denormalize(
    x, 
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)
def save_test_results(tensor, target, output_class, counter):
    """
    This function will save a few test images along with the 
    ground truth label and predicted label annotated on the image.
    :param tensor: The image tensor.
    :param target: The ground truth class number.
    :param output_class: The predicted class number.
    :param counter: The test image number.
    """
    image = denormalize(tensor).cpu()
    image = image.squeeze(0).permute((1, 2, 0)).numpy()
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gt = target.cpu().numpy()
    cv2.putText(
        image, f"GT: {CLASS_NAMES[int(gt)]}", 
        (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, (0, 255, 0), 2, cv2.LINE_AA
    )
    if output_class == gt:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv2.putText(
        image, f"Pred: {CLASS_NAMES[int(output_class)]}", 
        (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, color, 2, cv2.LINE_AA
    )
    cv2.imwrite(
        os.path.join('..', 'outputs', 'test_results', 'test_image_'+str(counter)+'.png'), 
        image*255.
    )
def test(model, testloader, DEVICE):
    """
    Function to test the trained model on the test dataset.
    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.
    Returns:
        predictions_list: List containing all the predicted class numbers.
        ground_truth_list: List containing all the ground truth class numbers.
        acc: The test accuracy.
    """
    model.eval()
    print('Testing model')
    predictions_list = []
    ground_truth_list = []
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)
            # Forward pass.
            outputs = model(image)
            # Softmax probabilities.
            predictions = F.softmax(outputs).cpu().numpy()
            # Predicted class number.
            output_class = np.argmax(predictions)
            # Append the GT and predictions to the respective lists.
            predictions_list.append(output_class)
            ground_truth_list.append(labels.cpu().numpy())
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()
            # Save a few test images.
            if counter % 99 == 0:
                save_test_results(image, labels, output_class, counter)
    acc = 100. * (test_running_correct / len(testloader.dataset))
    return predictions_list, ground_truth_list, acc
As we will be using ImageNet normalization stats while creating the datasets, so, we will also need to denormalize them. We need this for saving the image properly to disk again with the predicted and ground-truth annotations. The denormalize func