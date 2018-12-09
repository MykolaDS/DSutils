import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    music_train = pd.read_csv('~/vodafone/data/train_music.csv')
    music_train = music_train.drop(columns=['id'])
    target_train, features_train = music_train['target'], music_train.drop(columns=['target'])
    #
    music_test = pd.read_csv('~/vodafone/data/test_music.csv')
    ID = music_test['id']
    features_test = music_test.drop(columns=['id'])
    #
    features_train = features_train.fillna(features_train.max())
    #
    features_test = features_test.fillna(features_train.max())
    features_test = features_test.mask(np.isinf(features_test))
    features_test = features_test.fillna(features_train.max())

    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1)
    rf.fit(X=features_train,y=target_train)

    score = rf.predict_proba(features_test)

    output = pd.DataFrame()
    output['id'] = ID
    output['prediction'] = score[:,1]
    output.to_csv('rf_fitted_curve_max_fill.csv',index=False)


