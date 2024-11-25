import os

def create_eval_directories(path):
    datasets = ['cryonuseg', 'lynsec', 'lizard', 'pannuke', 'monusac', 'monuseg', 'tnbc', 'puma', 'jano', 'nuinsseg', 'cpm15', 'cpm17']
    for dataset in datasets:
        dataset_path = os.path.join(path, f'{dataset}_eval')
        for mode in ['instance', 'boxes', 'points', 'amg']:
            os.makedirs(os.path.join(dataset_path, mode), exist_ok=True)


create_eval_directories('/mnt/lustre-grete/usr/u12649/scratch/models/old_pannuke_sam_eval')
