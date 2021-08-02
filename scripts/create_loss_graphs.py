import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


def plot_metric(metric: str, trials: Dict[str, int], validation=False, smoothing=0.9, filename=None):
    master = os.getenv("DET_MASTER")
    login_data = {
        'username': os.getenv('DET_USERNAME'),
        'password': os.getenv('DET_PASSWORD')
    }
    login_response = requests.post(f'{master}/api/v1/auth/login', json=login_data).json()
    token = login_response['token']
    headers = {'Authorization': f'Bearer {token}'}

    assert token is not None

    data = []

    for name, trial in trials.items():
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
            'x': [workload['priorBatchesProcessed'] for workload in workloads],
            'y': [workload['metrics'][metric] for workload in workloads]
        })

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

    if filename:
        fig.savefig(f'graphs/{filename}', bbox_inches='tight', pad_inches=0.01)

    plt.show()


plot_metric(
    'd_loss',
    {
        'depth 64': 26478,
        'depth 32': 26477,
        'depth 16': 25637,
        'depth 8': 25635
    },
    filename='anime_face_d_loss.eps'
)

plot_metric(
    'd_loss',
    {
        'depth 64': 26476,
        'depth 32': 26475,
        'depth 16': 25634,
        'depth 8': 25632
    },
    filename='celeba_hq_d_loss.eps'
)
