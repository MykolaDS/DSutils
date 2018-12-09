import pandas as pd
import numpy as np
def average(*pathes):
    dataframes = [pd.read_csv(path) for path in pathes]
    predictions = [np.array(df['prediction']) for df in dataframes]

    predictions = np.stack(predictions)
    predictions_average = predictions.mean(axis=0)

    average_out = pd.DataFrame()
    average_out['id'] = dataframes[0]['id']
    average_out['prediction'] = predictions_average
    return average_out
if __name__ == '__main__':
    pathes = ['~/Downloads/catbooster.csv','~/Downloads/prediction_test_baseline_full_train_set_full_train_median_rf.csv']
    average_out = average(*pathes)
    average_out.to_csv('average_catboost_and_rf_baseline.csv',index=False)