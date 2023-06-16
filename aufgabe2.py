import os
from functools import reduce

import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

"""
Tutorial source: https://towardsdatascience.com/a-practical-guide-to-implementing-a-random-forest-classifier-in-python-979988d8a263
"""

# Aufgabe a
train_files = [f'Bilder/Flur/Flur{i + 1}.jpg' for i in range(20)] + \
              [f'Bilder/Labor/Labor{i + 1}.jpg' for i in range(20)] + \
              [f'Bilder/Professorenbuero/Professorenbuero{i + 1}.jpg' for i in range(20)] + \
              [f'Bilder/Teekueche/Teekueche{i + 1}.jpg' for i in range(20)]

test_files = [f'Bilder/Flur/Flur{i + 21}.jpg' for i in range(5)] + \
             [f'Bilder/Labor/Labor{i + 21}.jpg' for i in range(5)] + \
             [f'Bilder/Professorenbuero/Professorenbuero{i + 21}.jpg' for i in range(5)] + \
             [f'Bilder/Teekueche/Teekueche{i + 21}.jpg' for i in range(5)]

assert len(train_files) + len(test_files) == 100


# Aufgabe b
def get_histogram(file: str):
    """
    Returns the r, g, b histogram data for a file
    """
    img = Image.open(file)
    r, g, b = img.split()
    hist_r = r.histogram()
    hist_g = g.histogram()
    hist_b = b.histogram()

    return hist_r, hist_g, hist_b


def sum_histogram(folder: str):
    """
    Creates the histogram data for a folder of images
    """

    results_r = list()
    results_g = list()
    results_b = list()

    for file in os.listdir(folder):
        hist_r, hist_g, hist_b = get_histogram(f'{folder}/{file}')
        results_r.append(hist_r)
        results_g.append(hist_g)
        results_b.append(hist_b)

    return [sum(e) for e in zip(*results_r)], [sum(e) for e in zip(*results_g)], [sum(e) for e in zip(*results_b)]


def create_histograms():
    """
    Creates the histogram images
    """

    def save_plot(data, color, filename):
        plt.bar(range(1, 257), data, width=1, edgecolor='none', color=color)
        plt.xlim([-0.5, 255.5])
        plt.savefig(f'Histograms/{filename}')
        plt.close()

    os.makedirs('Histograms', exist_ok=True)

    for folder in ('Flur', 'Labor', 'Professorenbuero', 'Teekueche'):
        sum_r, sum_g, sum_b = sum_histogram(f'Bilder/{folder}')

        # Save histograms
        save_plot(sum_r, 'red', f'{folder}_r.png')
        save_plot(sum_g, 'green', f'{folder}_g.png')
        save_plot(sum_b, 'blue', f'{folder}_b.png')


# create_histograms()


""" Aufgabe c """
def preprocess_data(data: list):
    """
    Preprocesses the data for training / evaluation
    The training data is the r, g, b histograms in a list
    """

    raw_data = [get_histogram(f) for f in data]
    combine = lambda x, y: x + y
    data = [reduce(combine, e) for e in raw_data]

    return data

# Create the histograms for training / testing data
train_data = preprocess_data(train_files)
train_labels = [0] * 20 + [1] * 20 + [2] * 20 + [3] * 20
test_data = preprocess_data(test_files)
test_labels = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5
all_data = train_data + test_data
all_labels = train_labels + test_labels

rf = RandomForestClassifier()
rf.fit(train_data, train_labels)

# Calculate accuracy
train_accuracy = rf.score(train_data, train_labels)
print(f'Training accuracy: {train_accuracy}')
test_accuracy = rf.score(test_data, test_labels)
print(f'Testing accuracy: {test_accuracy}')

# Calculate F1-Score
f1_score = metrics.f1_score(all_labels, rf.predict(all_data), average='weighted')
print(f'F1-score: {f1_score}')

# Calculate precision
precision = metrics.precision_score(all_labels, rf.predict(all_data), average='weighted')
print(f'Precision: {precision}')

# Create confusion matrix
confusion_matrix = metrics.confusion_matrix(test_labels, rf.predict(test_data))
print(f'Confusion matrix:\n{confusion_matrix}')


print('Done.')
