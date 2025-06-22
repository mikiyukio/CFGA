
import numpy as np
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from os.path import join
import os
import csv
import sys

model_will_test = sys.argv[1]
dataset_name = sys.argv[2]
PARA=[]
PARA.append(model_will_test)

files_json=[
            'layer4_1_conv1_weight.json',
            'layer4_1_conv2_weight.json','layer4_2_conv2_weight.json',
            'layer4_conv2_weight.json',
            'layer4_1_conv3_weight.json',
            'layer4_2_conv3_weight.json',
            'layer4_2_conv1_weight.json',
            'embedding_max_avg.json',
            'scda_max_avg_norm.json',
            'scda_norm_max_avg.json',
            ]
results_csv=[
            'layer4_1_conv1_weight.csv',
            'layer4_1_conv2_weight.csv','layer4_2_conv2_weight.csv',
            'layer4_conv2_weight.csv',
            'layer4_1_conv3_weight.csv',
            'layer4_2_conv3_weight.csv',
            'layer4_2_conv1_weight.csv',
            'embedding_max_avg.csv',
            'scda_max_avg_norm.csv',
            'scda_norm_max_avg.csv',
            ]



class RetMetric():
    def __init__(self, feats, labels):

        if len(feats) == 2 and type(feats) == list:
            """
            feats = [gallery_feats, query_feats]
            labels = [gallery_labels, query_labels]
            """
            self.is_equal_query = False

            self.gallery_feats, self.query_feats = feats
            self.gallery_labels, self.query_labels = labels

        else:
            self.is_equal_query = True
            self.gallery_feats = self.query_feats = feats
            self.gallery_labels = self.query_labels = labels

        self.sim_mat = np.matmul(self.query_feats, np.transpose(self.gallery_feats))

    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0

        for i in range(m):
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]

            thresh = np.sort(pos_sim)[-2] if self.is_equal_query else np.max(pos_sim)

            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m






for index in range(len(PARA)):
    PARA_I = PARA[index].replace('.', '_')  # 避免路径出错
    target_path = join('datafile',dataset_name, PARA_I)
    for index_2 in range(len(files_json)):

        filename=join(target_path,files_json[index_2])
        print(filename)

        with open(filename) as f_obj:
            final_features=json.load(f_obj)
        # train_data=final_features['train']
        test_data=final_features['test']
        print('ok******************')


        train_paths_name='./datafile/{}/train_paths.json'.format(dataset_name)
        test_paths_name='./datafile/{}/test_paths.json'.format(dataset_name)
        train_labels_name='./datafile/{}/train_labels.json'.format(dataset_name)
        test_labels_name='./datafile/{}/test_labels.json'.format(dataset_name)
        with open(train_paths_name) as miki:
                train_paths=json.load(miki)
        with open(test_paths_name) as miki:
                test_paths=json.load(miki)
        with open(train_labels_name) as miki:
                train_labels=json.load(miki)
        with open(test_labels_name) as miki:
                test_labels=json.load(miki)


        ########################333
        # dataset_labels=test_labels+train_labels
        dataset_labels=np.array(test_labels)

        #############################
        X = np.array(test_data)
        print(X.shape)

        metric=RetMetric(X,dataset_labels)
        print(metric.recall_k(1))
        print(metric.recall_k(2))
        print(metric.recall_k(4))
        print(metric.recall_k(8))
        print(metric.recall_k(16))
        print(metric.recall_k(32))
