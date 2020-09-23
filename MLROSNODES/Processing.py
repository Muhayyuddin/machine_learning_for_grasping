import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import seaborn as sns
from matplotlib import pyplot as plt

data_path = os.getcwd() + '/CSV'
images_path = os.getcwd() + '/images'
print("Path : " + data_path)
lst = os.listdir(data_path)
print("List : ", lst)


def read_dataset(name='CSV/BanPreparedData.csv'):
    dataset = pd.read_csv(name, sep=',', encoding='utf-8')
    del (dataset['Unnamed: 0'])
    dataset = shuffle(dataset)
    return dataset


def split_dataset(dataset):
    train, test = train_test_split(dataset, test_size=0.20)
    return train, test


def data_shape(dataset):
    print('dataset columns names : ', dataset.columns)
    print('dataset shape is ', dataset.shape)


def check_balance(dataset):
    label = dataset.label.value_counts()
    plot = label.plot(kind='bar', x='y')
    plot.set_xlabel('y')
    plot.set_ylabel('Frequency')
    plot.set_title("Total number of 'yes' & 'no' in train DataSet")


# plot.show()


def describe_dataset(dataset):
    data = dataset.describe()
    data.transpose()


def label_data(dataset):
    encoded_dataset = dataset.copy()
    object_encoder = LabelEncoder()
    actions_encoder = LabelEncoder()
    label_encoder = LabelEncoder()
    encoded_dataset['object_name'] = object_encoder.fit_transform(dataset['object_name'])
    encoded_dataset['actions'] = actions_encoder.fit_transform(dataset['actions'])
    encoded_dataset['label'] = label_encoder.fit_transform(dataset['label'])
    return encoded_dataset, object_encoder, actions_encoder, label_encoder


def extract_feature_values(dataset):
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return x, y


def print_confusion_matrix(name, confusion_matrix, class_names, figure_size=(8, 5), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figure_size)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label', fontsize=22)
    plt.xlabel('Predicted label', fontsize=22)
    plt.savefig('images/' + name+'.jpeg', dpi=600, quality=100, format='jpeg', pad_inches=0.1,
                transparent=True, bbox_inches='tight')
    plt.show()


def draw_predicted_graph(prediction_model):
    plt.rc('ytick', labelsize=14)
    plot = prediction_model.plot.barh(x='Model', y='Accuracy', legend=None)
    plot.set_ylabel('Models', fontdict={'fontsize': 16, 'fontweight': 'heavy'})
    plot.set_xlabel('Accuracy', fontdict={'fontsize': 16, 'fontweight': 'heavy'})
    plot.set_title("Models and their Accuracy scores", fontdict={'fontsize': 16, 'fontweight': 'heavy'})
    fig = plot.get_figure()
    fig.savefig("images/ModelsAccuracy1.jpeg", dpi=600, format='jpeg', quality=100, pad_inches=0.1, transparent=True,
                bbox_inches='tight')
