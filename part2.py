import random
import time
import math
import pandas as pd
from collections import defaultdict

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

imdb_df = pd.read_csv(imdb_path) #, nrows=200)
imdb_df = imdb_df.drop_duplicates(subset=['Movie Name', 'Movie Date'], keep='first')
print(imdb_df.columns)
print(imdb_df.head)

import unicodedata
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
# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Build the category_lines dictionary, a list of descriptins per 
category_lines = {}
all_categories = []

# Read a file and split into lines
def get_descriptions():
    descriptions = list(imdb_df['Description'])
    return [unicodeToAscii(desc) for desc in descriptions]

def get_categories():
    movie_types = set() #imdb_df['Movie Type'])
    movie_type_to_descripton_mapping = defaultdict(set)
    for _, row in imdb_df.iterrows():
        row_movie_types = [mtyp.strip() for mtyp in row['Movie Type'].split(',')]
        movie_description = row['Description']
        for m_type in row_movie_types:
            movie_type_to_descripton_mapping[m_type].add(movie_description)
        movie_types.update(row_movie_types)

    n_categories = len(movie_types)

    return {key:list(val) for key, val in movie_type_to_descripton_mapping.items()}, list(movie_types), n_categories

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def descToTensor(description):
    tensor = torch.zeros(len(description), 1, n_letters)
    for li, letter in enumerate(description):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

"""
Create network
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
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(all_categories, category_descs):
    category = randomChoice(all_categories)
    line = randomChoice(category_descs[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = descToTensor(line)
    return category, line, category_tensor, line_tensor

#for i in range(10):
#    category, line, category_tensor, line_tensor = randomTrainingExample()
#    print('category =', category, '/ line =', line)


def train(category_tensor, line_tensor, rnn, learning_rate):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

if __name__ == '__main__':
    category_descs, all_categories, n_categories = get_categories()
    print(f"category lines keys: {category_descs.keys()}.\nCategory_descs subsample: {list(category_descs['Crime'])[:5]}\nCatgories subsample: {all_categories[:5]}, \nNo. of categories: {n_categories}")

    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    criterion = nn.NLLLoss()
    
    learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
    n_iters = 100000
    print_every = 10000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_descs)
        output, loss = train(category_tensor, line_tensor, rnn, learning_rate)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
    

    """
    Evaluation
    """
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        if i % 100:
            print(f"{i} iterations completed.")
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_descs)
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
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

    # sphinx_gallery_thumbnail_number = 2
    plt.show()