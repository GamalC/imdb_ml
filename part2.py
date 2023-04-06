import random
import time
import math
import logging
import sys
import pandas as pd
from collections import defaultdict

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

imdb_path = "imdb_newdataset/imdb_database.csv"

"""
Part 2
Your task is to train a classifier that can predict the correct movie types based on
the description (plus any other features if you find them useful). In your notebook
or script, please:
- first carry out any data preparation/cleaning that you think are
necessary and 
- then split your data into train/valid/test sets (in the interest of time
feel free to use a subset of the data). 
- Select a relevant model and report results on your test set.
"""

logging.info("Loading in dataset...")
imdb_df = pd.read_csv(imdb_path)
#Remove duplicates from the dataset
imdb_df = imdb_df.drop_duplicates(subset=['Movie Name', 'Movie Date'], keep='first')

import string
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

"""
Prepocessing.
"""
# Build the category_descriptions dictionary, a list of descriptins per 
category_descriptions = {}
all_categories = []

#Get dataset into format needed: link movie types to the descriptions and get categories.
def get_dataset():
    movie_types = set()
    movie_type_to_descripton_mapping = defaultdict(set)
    for _, row in imdb_df.iterrows():
        row_movie_types = [mtyp.strip() for mtyp in row['Movie Type'].split(',')]
        movie_description = row['Description']
        for m_type in row_movie_types:
            movie_type_to_descripton_mapping[m_type].add(movie_description)
        movie_types.update(row_movie_types)

    #Category has too few instances to spread across splits.
    #TODO: moe sophisticated way of detectingthese catagories and removing them.
    movie_types.remove('Adult')
    del movie_type_to_descripton_mapping['Adult']

    n_categories = len(movie_types)

    return {key:list(val) for key, val in movie_type_to_descripton_mapping.items()}, list(movie_types), n_categories

#Split the dataset into train/dev/test subsets.
#TODO: use more sophisticated datset objects and tools provided in libraruies like PyTorch.
def split_dataset(dataset_dict, dev_pct=10, test_pct=20):
    train_set = defaultdict(list)
    dev_set = defaultdict(list)
    test_set = defaultdict(list)
    dev_len = 0
    test_len, train_len = 0, 0 

    ds_instance_cnt = 0
    for _, vals in dataset_dict.items():
        ds_instance_cnt += len(vals)

    max_test_len = int(test_pct/100.0 * ds_instance_cnt)
    max_dev_len = int(dev_pct/100.0 * ds_instance_cnt)

    #Ensure all sets have at least one entry in each category
    for label, values in dataset_dict.items():
        for set_choice, desc_ in zip(['train', 'dev', 'test'], values[:3]):
            if set_choice == 'dev' and dev_len < max_dev_len:
                dev_set[label].append(desc_)
                dev_len += 1
            elif set_choice == 'test' and test_len < max_test_len:
                test_set[label].append(desc_)
                test_len += 1
            else:
                train_set[label].append(desc_)
                train_len += 1

    #Allocate remaining instances.
    for label, values in dataset_dict.items():
        for desc_ in values[3:]:
            set_choice = random.choices(['train', 'dev', 'test'], weights=(100-(dev_pct+test_pct), dev_pct, test_pct))[0]
            if set_choice == 'dev' and dev_len < max_dev_len:
                dev_set[label].append(desc_)
                dev_len += 1
            elif set_choice == 'test' and test_len < max_test_len:
                test_set[label].append(desc_)
                test_len += 1
            else:
                train_set[label].append(desc_)
                train_len += 1

    logging.info(f"Data split complete. Total instances was {ds_instance_cnt:,}.\nDatsest sizes: Train: {train_len:,}, dev: {dev_len:,}, test: {test_len:,}")
    assert len(train_set) == len(dev_set) == len(test_set), f"Unaligned amount of movie categories: {len(train_set)}, {len(dev_set)}, {len(test_set)}."

    return train_set, dev_set, test_set

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a movie description into a <description_char_length x 1 x n_letters>, or an array of one-hot letter vectors
def descToTensor(description):
    tensor = torch.zeros(len(description), 1, n_letters)
    for li, letter in enumerate(description):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

"""
Create neural network model. An RNN.
"""
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

"""
Training.
"""
#Return human readable catgeory given an output vector.
def categoryFromOutput(output):
    _, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

#Get a random training example
def randomTrainingExample(all_categories, category_descs):
    category = random.choice(all_categories)
    description = random.choice(category_descs[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    desc_tensor = descToTensor(description)
    return category, description, category_tensor, desc_tensor

#Train RNN model
def train(category_tensor, desc_tensor, rnn, learning_rate, criterion):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(desc_tensor.size()[0]):
        output, hidden = rnn(desc_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

#Return time elapsed since a given time
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train_setup(rnn, criterion, all_labels, train_set, dev_set):
    learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
    n_iters = 100
    print_every = 10

    # Keep track of losses for plotting
    current_loss = 0

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, description, category_tensor, desc_tensor = randomTrainingExample(all_labels, train_set)
        _, loss = train(category_tensor, desc_tensor, rnn, learning_rate, criterion)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            category, description, _, desc_tensor = randomTrainingExample(all_labels, dev_set)
            dev_output = evaluate(desc_tensor, rnn)
            guess, _ = categoryFromOutput(dev_output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            logging.info('%d %d%% (%s) %.4f %s. Result: Guess: %s Gold: %s' % (iter, iter / n_iters * 100, timeSince(start), loss, description, guess, correct))

"""
Evaluate model
"""
#Return an output given a description
def evaluate(desc_tensor, rnn):
    hidden = rnn.initHidden()

    for i in range(desc_tensor.size()[0]):
        output, hidden = rnn(desc_tensor[i], hidden)

    return output


def evaluate_with_confusion_matrix(rnn, all_categories, category_descs, n_categories):
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 100

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, _, _, desc_tensor = randomTrainingExample(all_categories, category_descs)
        output = evaluate(desc_tensor, rnn)
        _, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

if __name__ == '__main__':
    logging.info("Preprocessing the data....")
    category_descs, all_categories, n_categories = get_dataset()
    train_category_descs, dev_category_descs, test_category_descs = split_dataset(category_descs)
    
    logging.info(f"category descriptions keys: {category_descs.keys()}.\nCategory_descs subsample: {list(category_descs['Crime'])[:5]}\nCatgories subsample: {all_categories[:5]}, \nNo. of categories: {n_categories}")

    logging.info("Setting up the model...")
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)
    criterion = nn.NLLLoss()

    logging.info("Training the model...")
    train_setup(rnn, criterion, all_categories, train_category_descs, dev_category_descs)
    logging.info("Evaluating the model...")
    evaluate_with_confusion_matrix(rnn, all_categories, test_category_descs, n_categories)
