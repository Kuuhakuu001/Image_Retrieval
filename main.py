import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f"{ROOT}/train")))


def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)


def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path


def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)


def get_l1_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)  # numpy array of images and paths
            rates = absolute_difference(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def plot_results(query_path, ls_path_score, top_n=5, reverse=False):
    # Sort the results based on the L1 score
    ls_path_score.sort(key=lambda x: x[1], reverse=reverse)

    # Display the query image
    query_img = Image.open(query_path)
    plt.figure(figsize=(15, 5))

    # Plot the query image
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')

    # Plot the top-N similar images
    for i in range(top_n):
        img_path, score = ls_path_score[i]
        img = Image.open(img_path)
        plt.subplot(1, top_n + 1, i + 2)
        plt.imshow(img)
        plt.title(f"Score: {score:.2f}")
        plt.axis('off')

    plt.show()


def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.mean((data - query) ** 2, axis=axis_batch_size)


def get_l2_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + '/' + folder
            images_np, images_path = folder_to_images(path, size)  # numpy array of images and paths
            rates = mean_square_difference(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    return np.sum(data * query, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


def get_cosine_similarity_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + '/' + folder
            images_np, images_path = folder_to_images(path, size)  # numpy array of images and paths
            rates = cosine_similarity(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def correlation_coefficient(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)

    query_norm = np.sqrt(np.sum(query_mean ** 2))
    data_norm = np.sqrt(np.sum(data_mean ** 2, axis=axis_batch_size))

    return np.sum(data_mean * query_mean, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


def get_correlation_coefficient_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + '/' + folder
            images_np, images_path = folder_to_images(path, size)  # numpy array of images and paths
            rates = correlation_coefficient(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


# L1-Score for simple images

# root_img_path = f"{ROOT}/train/"
# query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
# size = (448, 448)
# query, ls_path_score = get_l1_score(root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, reverse=False)


# L1-Score for complex images
# root_img_path = f"{ROOT}/train/"
# query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
# size = (448, 448)
# query, ls_path_score = get_l1_score(root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, reverse=False)

# L2-Score for simple images
# root_img_path = f"{ROOT}/train/"
# query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
# size = (448, 448)
# query, ls_path_score = get_l2_score(root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, reverse=False)

# L2-Score for complex images
# root_img_path = f"{ROOT}/train/"
# query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
# size = (448, 448)
# query, ls_path_score = get_l2_score(root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, reverse=False)

# Cosine-similarities for simple images
# root_img_path = f"{ROOT}/train/"
# query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
# size = (448, 448)
# query, ls_path_score = get_cosine_similarity_score(root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, reverse=True)

# Cosine-similarities for complex images
# root_img_path = f"{ROOT}/train/"
# query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
# size = (448, 448)
# query, ls_path_score = get_cosine_similarity_score(root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, reverse=True)

# Correlation-Coefficient for simple images
# root_img_path = f"{ROOT}/train/"
# query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
# size = (448, 448)
# query, ls_path_score = get_correlation_coefficient_score(root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, reverse=True)

# Correlation-Coefficient for complex images
root_img_path = f"{ROOT}/train/"
query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
size = (448, 448)
query, ls_path_score = get_correlation_coefficient_score(root_img_path, query_path, size)
plot_results(query_path, ls_path_score, reverse=True)
