import os

from determined.experimental import client
from dotenv import load_dotenv

load_dotenv()

client.login(
    master=os.getenv('DET_MASTER'),
    user=os.getenv('DET_USERNAME'),
    password=os.getenv('DET_PASSWORD')
)


def download_and_organize_checkpoints(trial_id, name, path='./checkpoints'):
    trial = client.get_trial(trial_id)
    checkpoint_best = trial.select_checkpoint(best=True)
    checkpoint_latest = trial.select_checkpoint(latest=True)

    checkpoint_best.download(path=f'{path}/{name}_best')
    print(f'Trial {trial_id} -> {name}_best finished!')
    checkpoint_latest.download(path=f'{path}/{name}_latest')
    print(f'Trial {trial_id} -> {name}_latest finished!')


experiments = [
    ('01_batch_normalization', 76235),
    ('02_batch_normalization_ema', 76234),
    ('03_spectral_normalization', 76305),
    ('04_spectral_normalization_ema', 76301),
    ('05_layer_normalization', 76232),
    ('06_layer_normalization_ema', 76233),
    ('07_no_normalization', 76336),
    ('08_no_normalization_ema', 76335),
]

for name, trial_id in experiments:
    download_and_organize_checkpoints(trial_id, name)
