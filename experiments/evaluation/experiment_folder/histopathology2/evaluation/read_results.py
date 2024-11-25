import pandas as pd
import os
from natsort import natsorted
model_name = 'pannuke'
eval_path = os.path.join(f'/mnt/lustre-grete/usr/u12649/scratch/models/{model_name}_sam_eval')


def read_instance_csv(path):
    result_dict = {
        'dataset':[],
        'msa':[],
        'sa50':[],
        'sa75':[]
    }
    for dataset in natsorted(['pannuke','lynsec', 'cryonuseg', 'lizard', 'tnbc', 'monusac', 'monuseg', 'puma', 'jano', 'cpm15', 'cpm17', 'nuinsseg']):  
        dataset_path = os.path.join(path, f'{dataset}_eval', 'instance/results/instance_segmentation_with_decoder.csv')
        if not os.path.exists(dataset_path):
            continue
        df = pd.read_csv(dataset_path)
        result_dict['msa'].append(df.loc[0, 'mSA'])
        result_dict['sa50'].append(df.loc[0, 'SA50'])
        result_dict['sa75'].append(df.loc[0, 'SA75'])
        result_dict['dataset'].append(dataset)
    df = pd.DataFrame(result_dict) 
    print('Results of {instance segmentation evaluation:')
    print(df.head(12))
    csv_path = os.path.join(eval_path, 'instance_results.csv')
    df.to_csv(csv_path, index=False)


read_instance_csv(eval_path)


def read_amg_csv(path):
    result_dict = {
        'dataset':[],
        'msa':[],
        'sa50':[],
        'sa75':[]
    }
    for dataset in natsorted(['pannuke','lynsec', 'cryonuseg', 'lizard', 'tnbc', 'monusac', 'monuseg', 'puma', 'jano', 'cpm15', 'cpm17', 'nuinsseg']):  
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
    csv_path = os.path.join(eval_path, 'amg_results.csv')
    df.to_csv(csv_path, index=False)


read_amg_csv(eval_path)


def read_it_boxes_csv(path):
    result_dict = {
        'dataset':[],
        'msa_1st':[],
        'msa_8th':[],
        'sa50_1st':[],
        'sa50_8th':[],
        'sa75_1st':[],
        'sa75_8th':[]
    }
    for dataset in natsorted(['pannuke','lynsec', 'cryonuseg', 'lizard', 'tnbc', 'monusac', 'monuseg', 'puma', 'jano', 'cpm15', 'cpm17', 'nuinsseg']):  
        dataset_path = os.path.join(path, f'{dataset}_eval', 'boxes/results/iterative_prompting_without_mask/iterative_prompts_start_box.csv')
        if not os.path.exists(dataset_path):
            continue
        df = pd.read_csv(dataset_path)
        # print(df.head(8))
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
    csv_path = os.path.join(eval_path, 'boxes_results.csv')
    df.to_csv(csv_path, index=False)

read_it_boxes_csv(eval_path)


def read_it_points_csv(path):
    result_dict = {
        'dataset':[],
        'msa_1st':[],
        'msa_8th':[],
        'sa50_1st':[],
        'sa50_8th':[],
        'sa75_1st':[],
        'sa75_8th':[]
    }
    for dataset in natsorted(['pannuke','lynsec', 'cryonuseg', 'lizard', 'tnbc', 'monusac', 'monuseg', 'puma', 'jano', 'cpm15', 'cpm17', 'nuinsseg']):  
        dataset_path = os.path.join(path, f'{dataset}_eval', 'points/results/iterative_prompting_without_mask/iterative_prompts_start_point.csv')
        if not os.path.exists(dataset_path):
            continue
        df = pd.read_csv(dataset_path)
        # print(df.head(8))
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
    csv_path = os.path.join(eval_path, 'instance_results.csv')
    df.to_csv(csv_path, index=False)


read_it_points_csv(eval_path)
