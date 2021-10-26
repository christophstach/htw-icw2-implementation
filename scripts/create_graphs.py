import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


def plot_graph(graphs: List[Tuple[str, int, bool, str]], filename: str = None, smoothing=0.9):
    master = os.getenv('DET_MASTER')
    login_data = {
        'username': os.getenv('DET_USERNAME'),
        'password': os.getenv('DET_PASSWORD')
    }
    login_response = requests.post(f'{master}/api/v1/auth/login', json=login_data).json()
    token = login_response['token']
    headers = {'Authorization': f'Bearer {token}'}

    assert token is not None

    data = []

    for name, trial, validation, metric in graphs:
        trial_response = requests.get(f'{master}/api/v1/trials/{trial}', headers=headers).json()

        if validation:
            workloads = [
                workload['validation']
                for workload
                in trial_response['workloads']
                if 'validation' in workload.keys()
                   and workload['validation']['state'] == 'STATE_COMPLETED'
            ]
        else:
            workloads = [
                workload['training']
                for workload
                in trial_response['workloads']
                if 'training' in workload.keys()
                   and workload['training']['state'] == 'STATE_COMPLETED'
            ]

        data.append({
            'name': name,
            'x': [workload['totalBatches'] for workload in workloads],
            'y': [workload['metrics'][metric] for workload in workloads]
        })

        print('.', end='')

    print()

    fig = plt.figure(figsize=(6, 6))
    fig.tight_layout(pad=0.01)

    ax = fig.add_subplot(1, 1, 1)

    for trial in data:
        y = pd.DataFrame(trial['y']).ewm(alpha=1 - smoothing).mean().values
        ax.plot(trial['x'], y, linewidth=2, label=trial['name'])

    ax.legend()
    ax.grid(False)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_ylabel('Classifier Score')
    ax.set_xlabel('Batches Processed')
    ax.set_ylim([5.0, 9.0])

    if filename:
        fig.savefig(f'graphs/{filename}', bbox_inches='tight', pad_inches=0.01)

    plt.show()


plot_graph([
    ('train without EMA', 76235, False, 'classifier_score'),
    ('train with EMA', 76234, False, 'classifier_score'),
    ('validation without EMA', 76235, True, 'val_classifier_score'),
    ('validation with EMA', 76234, True, 'val_classifier_score')
], 'batch_normalization.eps')

plot_graph([
    ('train without EMA', 76305, False, 'classifier_score'),
    ('train with EMA', 76301, False, 'classifier_score'),
    ('validation without EMA', 76305, True, 'val_classifier_score'),
    ('validation with EMA', 76301, True, 'val_classifier_score')
], 'spectral_normalization.eps')

plot_graph([
    ('train without EMA', 76232, False, 'classifier_score'),
    ('train with EMA', 76233, False, 'classifier_score'),
    ('validation without EMA', 76232, True, 'val_classifier_score'),
    ('validation with EMA', 76233, True, 'val_classifier_score')
], 'layer_normalization.eps')

plot_graph([
    ('train without EMA', 76336, False, 'classifier_score'),
    ('train with EMA', 76335, False, 'classifier_score'),
    ('validation without EMA', 76336, True, 'val_classifier_score'),
    ('validation with EMA', 76335, True, 'val_classifier_score')
], 'none.eps')

###

plot_graph([
    ('Batch Normalization', 76235, False, 'classifier_score'),
    ('Batch Normalization with EMA', 76234, False, 'classifier_score'),
    ('Spectral Normalization', 76305, False, 'classifier_score'),
    ('Spectral Normalization with EMA', 76301, False, 'classifier_score'),
    ('Layer Normalization', 76232, False, 'classifier_score'),
    ('Layer Normalization with EMA', 76233, False, 'classifier_score'),
    ('No Normalization', 76336, False, 'classifier_score'),
    ('No Normalization with EMA', 76335, False, 'classifier_score'),
], 'all_train.eps')

plot_graph([
    ('Batch Normalization', 76235, True, 'val_classifier_score'),
    ('Batch Normalization with EMA', 76234, True, 'val_classifier_score'),
    ('Spectral Normalization', 76305, True, 'val_classifier_score'),
    ('Spectral Normalization with EMA', 76301, True, 'val_classifier_score'),
    ('Layer Normalization', 76232, True, 'val_classifier_score'),
    ('Layer Normalization with EMA', 76233, True, 'val_classifier_score'),
    ('No Normalization', 76336, True, 'val_classifier_score'),
    ('No Normalization with EMA', 76335, True, 'val_classifier_score'),
], 'all_validation.eps')
