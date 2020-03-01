import os
import json
import pandas as pd


def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


def parse_training_metrics(runs_dir=None, csv_dir=None, sep='|'):
    folder_list = [o for o in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, o)) and o != 'sources']
    metrics_dict = {}
    for folder in folder_list:
        with open(os.path.join(runs_dir, folder, 'metrics.json')) as json_file:
            data = json.load(json_file)
        metrics_dict[folder] = {metric: max(data[metric]['values']) for metric in data.keys()}
    metrics_df = pd.DataFrame(metrics_dict).T
    if not csv_dir:
        print("csv path not provided.")
    else:
        metrics_df.to_csv(csv_dir, sep=sep)
    metrics_df = metrics_df.style.apply(highlight_max)
    return metrics_df
