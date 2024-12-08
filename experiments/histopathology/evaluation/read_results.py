import os
import pandas as pd
from natsort import natsorted


MODEL_NAME = 'pannuke_sam'
EVAL_PATH = '/mnt/lustre-grete/usr/u12649/scratch/models/'
DATASETS = [
    'pannuke', 'lynsec', 'cryonuseg', 'lizard', 'tnbc', 'monusac',
    'monuseg', 'puma', 'jano', 'cpm15', 'cpm17', 'nuinsseg'
]


def read_instance_csv(path, model_name):
    eval_path = os.path.join(path, f'{model_name}_eval')
    if model_name == 'vanilla':
        return
    result_dict = {'dataset': [], 'msa': [], 'sa50': [], 'sa75': []}
    for dataset in natsorted(DATASETS):
        dataset_path = os.path.join(path, f'{dataset}_eval', 'instance/results/instance_segmentation_with_decoder.csv')
        if not os.path.exists(dataset_path):
            continue

        df = pd.read_csv(dataset_path)
        result_dict['msa'].append(df.loc[0, 'mSA'])
        result_dict['sa50'].append(df.loc[0, 'SA50'])
        result_dict['sa75'].append(df.loc[0, 'SA75'])
        result_dict['dataset'].append(dataset)

    df = pd.DataFrame(result_dict)
    print('Results of instance segmentation evaluation:')
    print(df.head(12))
    csv_path = os.path.join(eval_path, f'instance_results_{model_name}.csv')
    df.to_csv(csv_path, index=False)
    return df


def read_amg_csv(path, model_name=None):
    eval_path = os.path.join(f'/mnt/lustre-grete/usr/u12649/scratch/models/{model_name}_eval')
    result_dict = {'dataset': [], 'msa': [], 'sa50': [], 'sa75': []}
    for dataset in natsorted(DATASETS):
        dataset_path = os.path.join(path, f'{dataset}_eval', 'amg/results/amg.csv')
        if not os.path.exists(dataset_path):
            continue

        df = pd.read_csv(dataset_path)
        result_dict['msa'].append(df.loc[0, 'mSA'])
        result_dict['sa50'].append(df.loc[0, 'SA50'])
        result_dict['sa75'].append(df.loc[0, 'SA75'])
        result_dict['dataset'].append(dataset)

    df = pd.DataFrame(result_dict)
    print('Results of amg evaluation:')
    print(df.head(12))
    csv_path = os.path.join(eval_path, f'amg_results_{model_name}_sam.csv')
    df.to_csv(csv_path, index=False)
    return df


def read_it_boxes_csv(path, model_name=None):
    eval_path = os.path.join(f'/mnt/lustre-grete/usr/u12649/scratch/models/{model_name}_eval')
    result_dict = {
        'dataset': [], 'msa_1st': [], 'msa_8th': [], 'sa50_1st': [], 'sa50_8th': [], 'sa75_1st': [], 'sa75_8th': []
    }
    for dataset in natsorted(DATASETS):
        dataset_path = os.path.join(
            path, f'{dataset}_eval', 'boxes/results/iterative_prompting_without_mask/iterative_prompts_start_box.csv'
        )
        if not os.path.exists(dataset_path):
            continue

        df = pd.read_csv(dataset_path)
        result_dict['msa_1st'].append(df.loc[0, 'mSA'])
        result_dict['sa50_1st'].append(df.loc[0, 'SA50'])
        result_dict['sa75_1st'].append(df.loc[0, 'SA75'])
        result_dict['msa_8th'].append(df.loc[7, 'mSA'])
        result_dict['sa50_8th'].append(df.loc[7, 'SA50'])
        result_dict['sa75_8th'].append(df.loc[7, 'SA75'])
        result_dict['dataset'].append(dataset)

    df = pd.DataFrame(result_dict)
    print('Results of iterative prompting with boxes evaluation:')
    print(df.head(12))
    csv_path = os.path.join(eval_path, f'boxes_results_{model_name}.csv')
    df.to_csv(csv_path, index=False)
    return df


def read_it_points_csv(path, model_name=None):
    eval_path = os.path.join(f'/mnt/lustre-grete/usr/u12649/scratch/models/{model_name}_eval')
    result_dict = {
        'dataset': [], 'msa_1st': [], 'msa_8th': [], 'sa50_1st': [], 'sa50_8th': [], 'sa75_1st': [], 'sa75_8th': []
    }
    for dataset in natsorted(DATASETS):
        dataset_path = os.path.join(
            path, f'{dataset}_eval', 'points/results/iterative_prompting_without_mask/iterative_prompts_start_point.csv'
        )
        if not os.path.exists(dataset_path):
            continue

        df = pd.read_csv(dataset_path)
        result_dict['msa_1st'].append(df.loc[0, 'mSA'])
        result_dict['sa50_1st'].append(df.loc[0, 'SA50'])
        result_dict['sa75_1st'].append(df.loc[0, 'SA75'])
        result_dict['msa_8th'].append(df.loc[7, 'mSA'])
        result_dict['sa50_8th'].append(df.loc[7, 'SA50'])
        result_dict['sa75_8th'].append(df.loc[7, 'SA75'])
        result_dict['dataset'].append(dataset)

    df = pd.DataFrame(result_dict)
    print('Results of iterative prompting with points evaluation:')
    print(df.head(12))
    csv_path = os.path.join(eval_path, f'points_results_{model_name}.csv')
    df.to_csv(csv_path, index=False)
    return df


def get_comparison_csv(mode):
    if mode == 'points':
        vanilla_dataframe = read_it_points_csv(
            os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/vanilla_sam_eval'), 'vanilla_sam'
        )
        pannuke_dataframe = read_it_points_csv(
            os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam_eval'), 'pannuke_sam'
        )
        combined_df = pd.concat([vanilla_dataframe, pannuke_dataframe], axis=1)
        combined_df.to_csv('~/comb/combined_data_points.csv', index=False)
        return
    elif mode == 'boxes':
        vanilla_dataframe = read_it_boxes_csv(
            os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/vanilla_sam_eval'), 'vanilla_sam'
        )
        pannuke_dataframe = read_it_boxes_csv(
            os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam_eval'), 'pannuke_sam'
        )
        combined_df = pd.concat([vanilla_dataframe, pannuke_dataframe], axis=1)
        combined_df.to_csv('~/comb/combined_data_boxes.csv', index=False)
        return
    else:
        vanilla_dataframe = read_amg_csv(
            os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/vanilla_sam_eval'), 'vanilla_sam'
        )
        pannuke_dataframe_amg = read_amg_csv(
            os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam_eval'), 'pannuke_sam'
        )
        pannuke_dataframe_instance = read_instance_csv(
            os.path.join('/mnt/lustre-grete/usr/u12649/scratch/models/pannuke_sam_eval'), 'pannuke_sam'
        )
        combined_df = pd.concat([vanilla_dataframe, pannuke_dataframe_amg, pannuke_dataframe_instance], axis=1)
        combined_df.to_csv('~/comb/combined_data_automatic.csv', index=False)


def main():
    read_instance_csv(EVAL_PATH, MODEL_NAME)
    read_amg_csv(EVAL_PATH, MODEL_NAME)

    # read_it_boxes_csv(eval_path, model_name)
    # read_it_points_csv(eval_path)
    # get_comparison_csv('boxes')


if __name__ == "__main__":
    main()
