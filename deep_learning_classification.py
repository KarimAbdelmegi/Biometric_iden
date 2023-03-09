import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import math
import os
from sklearn.model_selection  import train_test_split
from collections import defaultdict
import tensorboard
from torch.utils.tensorboard import SummaryWriter


path = "segmented_ppg2"
csv_name = []
data_read = []
features = []
labels = []

"""
data1 = pd.read_csv("segmented_apg/subject1_sit.csv")
data2 = pd.read_csv("segmented_apg/subject2_sit.csv")
data3 = pd.read_csv("segmented_apg/subject3_sit.csv")
data4 = pd.read_csv("segmented_apg/subject4_sit.csv")
"""

class BiometricIdentifier(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size):
        super(BiometricIdentifier, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, embed_size),
                                     #nn.Tanh()
                                     nn.Sigmoid()
                                     )


    def forward(self, x):
        return self.network(x)


def replace_nan(data_):
    for i in range(len(data_)):
        row = data_.loc[[i]]
        data_.loc[[i]] = np.where(np.isnan(row), 0, row)


def labeling(data_, subject, limit):
    replace_nan(data_)

    #data_values = np.zeros((limit, 3000)) #for segmented_apg
    #data_values[:data_.shape[0], :data_.shape[1]] = data_.values[0 : limit, :] #for semented_apg

    data_values = np.zeros((limit, data_.shape[1])) #for segmented_apg2
    data_values[:data_.shape[0], :data_.shape[1]] = data_.values[0 : limit, :] #for segmented_apg2

    #for i in range(len(data_)):
     #   for j in range(len(data_.iloc[:,i])):
      #      data_values.iloc[i][j] = data_.iloc[i][j]

    #labels_ = np.full(limit, subject) #for segmented_apg
    labels_ = np.full(limit, subject)

    features_ = torch.tensor(data_values, dtype=torch.float32)
    labels_ = torch.tensor(labels_, dtype=torch.long)

    return features_, labels_


def pairwise_distances(embeddings, squared=False):
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))
    square_norm = torch.diag(dot_product)

    distances = torch.unsqueeze(square_norm, 0) - 2.0 * dot_product + torch.unsqueeze(square_norm, 1)
    distances = torch.maximum(distances, torch.tensor(0.0))

    if not squared:
        mask = torch.as_tensor(torch.eq(distances, torch.as_tensor(0.0)), dtype=torch.float)
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances.float())

        distances = distances * (1.0 - mask)
    return distances


def get_anchor_positive_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label
    indices_equal = torch.as_tensor(torch.eye(labels.shape[0]), dtype=torch.bool)
    indices_not_equal = torch.logical_not(indices_equal)

    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

    mask = torch.logical_and(labels_equal, indices_not_equal).float()
    return mask


def get_anchor_negative_mask(labels):

    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    mask = torch.logical_not(labels_equal)

    return mask


def get_triplet_mask(labels):
    # (a, p, n) equivalent to (i, j, k)

    indices_equal = torch.as_tensor(torch.eye(labels.shape[0]), dtype=torch.bool)
    indices_not_equal = torch.logical_not(indices_equal)

    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)

    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    mask = torch.logical_and(distinct_indices,valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    # Generate all the valid triplets and average the loss over the positive ones
    # Returns: triple_loss-> scalar tensor containing the triplet loss
    pairwise_dist = pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = get_triplet_mask(labels)
    mask = mask.float()

    triplet_loss = torch.multiply(mask, triplet_loss)

    triplet_loss = torch.maximum(triplet_loss, torch.tensor(0.0))
    valid_triplets = torch.greater(triplet_loss, 1e-16).float()

    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss
           #fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):

    pairwise_dist = pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = get_anchor_positive_mask(labels).float()

    anchor_positive_dist = torch.multiply(pairwise_dist, mask_anchor_positive)

    hardest_positive_dist = torch.max(anchor_positive_dist, 1, keepdim=True)

    mask_anchor_negative = get_anchor_negative_mask(labels).float()

    max_anchor_negative_dist = torch.max(pairwise_dist, 1, keepdim=True)

    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist.values * (1.0 - mask_anchor_negative)

    hardest_negative_dist = torch.min(anchor_negative_dist, 1, keepdim=True)

    triplet_loss = torch.maximum(hardest_positive_dist.values - hardest_negative_dist.values + margin, torch.tensor(0))

    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss


for filename in os.listdir(path):
    f = os.path.join(path, filename)
    if os.path.isfile(f):
        csv_name.append(f)

samples_per_subj = 50
for i , read in enumerate(csv_name):
    data_read = pd.read_csv(read)
    feature, label = labeling(data_read, i+1, samples_per_subj)
    features.extend(feature)
    labels.extend(label)


features = torch.stack(features)
labels = torch.stack(labels)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

"""
features, labels = labeling(data1, 1, 300)
features2, labels2 = labeling(data2, 2, 300)
features3, labels3 = labeling(data3, 3, 300)
features4, labels4 = labeling(data4, 4, 300)

features = torch.cat(( features ,features2, features3, features4), dim=0)
labels = torch.cat(( labels, labels2, labels3, labels4), dim=0)
"""
batch_size = 300

training_dataset = torch.utils.data.TensorDataset(features, labels)

#training_dataset = torch.utils.data.TensorDataset(features_train, labels_train)
#testing_dataset = torch.utils.data.TensorDataset(features_test, labels_test)


train_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
#test_data_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = BiometricIdentifier(input_size=3000, hidden_size=2000, embed_size=1000).to(device)
model = BiometricIdentifier(input_size=98, hidden_size=200, embed_size=64).to(device)
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1500
#total_samples = len(dataset)
#n_iterations = math.ceil(total_samples/batch_size)

loss_track = []

_feat, _lab = next(iter(train_data_loader))

writer = SummaryWriter(f'runs/subj {len(csv_name)} samples {samples_per_subj} batch {batch_size} LR {learning_rate}')
writer.add_graph(model, _feat)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for i, (feat, lab) in enumerate(train_data_loader):

        embeddings = model(feat.to(device))

        loss = batch_all_triplet_loss(lab.to(device), embeddings, margin=1, squared=False)

        writer.add_scalar('Loss/train', loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if epoch % 5 == 0:
            #print(f'Epoch {epoch}: Loss = {loss:.2f}')
            loss_track.append(loss.detach().numpy())

    print("Epoch: %d, loss: %.4f" % (epoch, total_loss/(i+1)))


writer.flush()

subjects = list(range(1,len(csv_name) + 1))
dataset_directory = defaultdict(list)

sub_avg_embed = {}


with torch.no_grad():
    for subject in subjects:
        for j in range(len(training_dataset)):
            feat, lab = training_dataset[j]
            if lab == subject:
                print(lab)
                if feat.min() != 0 and feat.max() != 0:
                    dataset_directory[subject].append(model(feat).to(device).detach().numpy())

    for sub in dataset_directory:
        sub_avg_embed[sub] = np.mean(dataset_directory[sub], 0)

    #model.eval()
    true_pos = false_neg = true_neg = false_neg = false_pos = 0
    dictionary = {}
    n_correct = 0
    n_samples = 0
    test_dictionary = {}
    training_dictionary = {}

    compare_this_dict = {}
    to_this_dict = {}

    for subject in subjects:
        for i in range(len(training_dataset)):
            feat, lab = training_dataset[i]
            if lab == subject:
                print(lab)
                if feat.min() != 0 and feat.max() != 0:
                    compare_this_dict[subject] = feat
                    break

    for person in sub_avg_embed:
        for i in range(1, len(subjects) + 1):
            compare1 = model(compare_this_dict[i].to(device))
            compare2 = torch.tensor(sub_avg_embed[person])
            cos = torch.dot(compare1, compare2) / (torch.norm(compare1) * torch.norm(compare2))
            print(cos)

            if person == i:
                if cos > 0.96:
                    true_pos += 1
                else:
                    false_neg += 1

            else:
                if cos > 0.96:
                    false_pos += 1
                    print('subject: ', i, 'is identical to ', person)
                else:
                    true_neg += 1

            n_samples += 1
    acc = (true_pos + true_neg) / n_samples
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print('Accuracy', acc)

''''
    # for testing on data seen by the model
    for feat, lab in training_dataset:
        if feat.min() != 0 and feat.max() != 0:
            lab.to(device)
            training_dictionary[lab.item()] = model(feat.to(device))
            print(lab)

    for person in training_dictionary:
        try:
            compare = model(dataset_directory[person].to(device))
            cos = torch.dot(training_dictionary[person], compare) / (torch.norm(training_dictionary[person]) * torch.norm(compare))
            print("Subject: ", person)

            if cos > 0.95:
                n_correct += 1


            print(cos)
            n_samples += 1

        except Exception as e:
            print("subject not in the database")



    # for testing on data not seen by the model
    for feat, lab in testing_dataset:
        if feat.min() != 0 and feat.max() != 0:
            lab.to(device)
            test_dictionary[lab.item()] = model(feat.to(device))

    for person in test_dictionary:
        try:
            compare = model(dataset_directory[person].to(device))
            cos = torch.dot(test_dictionary[person], compare) / (torch.norm(test_dictionary[person]) * torch.norm(compare))
            print("Subject: ", person)

            if cos > 0.9:
                n_correct += 1

            print(cos)
            n_samples += 1

        except Exception as e:
            print("subject not in the database")

    acc = n_correct/n_samples
    print(acc)


plt.plot(loss_track)
plt.show()
print("done")

'''