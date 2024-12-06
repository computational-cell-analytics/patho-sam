import shutil
from scipy.io import loadmat
import os
def postprocess_hovernet(output_path)
    for dataset in ['cpm15', 'cpm17', 'cryonuseg', 'janowczyk', 'lizard', 'lynsec', 'monusac', 'monuseg', 'nuinsseg', 'pannuke', 'puma', 'tnbc']:
        for model in ['consep', 'cpm17', 'kumar', 'pannuke', 'monusac']:
            output_path = os.path.join(output_dir, dataset, model)

            mat_to_tiff(os.path.join(output_path, 'mat'))
            shutil.rmtree(os.path.join(output_path, 'json'))
            shutil.rmtree(os.path.join(output_path, 'overlay'))


        