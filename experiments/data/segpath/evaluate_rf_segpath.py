import joblib
from pathlib import Path
import json
from sklearn.metrics import classification_report
from train_rf_segpath import get_rf_data


def evaluate_rf_data(path, rf_path, cell_types):
    rf_path = Path(rf_path)
    data_path = ""
    test_features, test_labels = get_rf_data(data_path, cell_types)
    rf = joblib.load(rf_path)
    pred = rf.predict(test_features)
    result_dict = classification_report(test_labels, pred, output_dict=True)
    json_output_path = path / f"evaluation_{rf_path.stem}.json"
    with open(json_output_path, 'a') as f:
        json.dump(result_dict, f, indent=4)
