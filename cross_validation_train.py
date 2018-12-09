from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json

def data_transfrom_train_baseline(data):
    target, features = data['target'], data.drop(columns=['target','id'])
    features_train_fill_by_median = features.fillna(features.median())
    return features_train_fill_by_median,target
def cross_validation_search(model_init,data,data_transform,params_lists,save_statistics=None,n_splits=10,shuffle=True,random_state=1):
    #
    features,target = data_transform(data)
    #
    kf = KFold(n_splits=n_splits,shuffle=shuffle,random_state=random_state)
    #
    scorings = []
    if not save_statistics:
        save_statistics = lambda x:x
    for param_list in params_lists:
        print(param_list)
        model = model_init(**param_list)
        curr_score = []
        start = time.time()
        for train_index, test_index in kf.split(features):
            features_train = features.loc[train_index,:]
            target_train = target[train_index]
            #
            features_test = features.loc[test_index, :]
            target_test = target[test_index]
            #
            model.fit(X=features_train,y=target_train)
            #
            probability_test = model.predict_proba(features_test)[:, 1]
            auc_score = roc_auc_score(y_true=target_test,y_score=probability_test)
            curr_score.append(auc_score)
        end = time.time()
        cur_statistics = save_statistics(curr_score)
        scorings.append((model,cur_statistics,param_list,end-start))
    return scorings

def main():
    kf = KFold(n_splits=10,shuffle=True,random_state=1)
    music_train = pd.read_csv('~/vodafone/data/train_music.csv')
    music_train = music_train.drop(columns=['id'])
    target_train,features_train = music_train['target'],music_train.drop(columns=['target'])
    features_train_fill_by_median = features_train.fillna(features_train.median())
    n_estimators_score_n_folds = []
    folds = [[(features_train_fill_by_median.loc[train_index,:],target_train.loc[train_index]),(features_train_fill_by_median.loc[test_index,:],target_train.loc[test_index])] for train_index,test_index in kf.split(features_train)]
    #
    for n_estimators in range(2,150):
        print(n_estimators)
        start = time.time()
        rf = RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1)
        roc_auc_list = []
        for train_index,test_index in kf.split(features_train):
            features_for_train = features_train_fill_by_median.loc[train_index,:]
            target_for_train = target_train.loc[train_index]
            #
            features_for_test = features_train_fill_by_median.loc[test_index,:]
            target_for_test = target_train.loc[test_index]
            #
            rf.fit(X=features_for_train,y=target_for_train)
            probability_test = rf.predict_proba(features_for_test)[:,1]
            auc_score = roc_auc_score(y_true=target_for_test,y_score=probability_test)
            roc_auc_list.append(auc_score)
        mean = np.mean(roc_auc_list)
        # varience = np.sqrt(np.var(roc_auc_list,ddof=1))
        # sqrt_n = len(roc_auc_list)
        end = time.time()
        n_estimators_score_n_folds.append((n_estimators,mean,end-start))
        print(end-start)
    n_estimators_list, means_list,_ = zip(*n_estimators_score_n_folds)
    plt.plot(n_estimators_list,means_list)
    plt.show()
    json.dump(n_estimators_score_n_folds,open('test.json','w'))
if __name__ == '__main__':
    music_train = pd.read_csv('~/vodafone/data/train_music.csv')
    params_lists = [{'n_estimators':n_estimators} for n_estimators in range(1,10)]
    scorings = cross_validation_search(RandomForestClassifier,music_train,data_transfrom_train_baseline,params_lists,save_statistics=np.mean)
    _,statistics,params_lists,_ = zip(*scorings)
    n_estimators = [param_list['n_estimators'] for param_list in params_lists]
    plt.plot(n_estimators,statistics)
    plt.show()
