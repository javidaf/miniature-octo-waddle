from ml_p3.utils.utils import prepare_cell_data, load_neural_network
import numpy as np
from astroglial_analysis.classifier2 import visualize_classifications

import matplotlib.pyplot as plt
import os


def predict_and_visualize(model, data_set, mask_file, classification_file):
    collumns = [0, 3, 4, 5, 6, 7, 8, 9, 10]

    data_features, _, _ = prepare_cell_data(data_set, collumns)

    predictions = model.predict_classes(data_features) + 1

    test_file = mask_file
    test_classifications_file = classification_file
    test_masks = np.load(test_file, allow_pickle=True).item()["masks"]
    test_classifications = np.load(test_classifications_file, allow_pickle=True).item()[
        "classifications"
    ]

    new_classifications = []
    for i in range(len(test_classifications)):
        classification = predictions[i]
        cell_label = test_classifications[i][1]
        new_classifications.append((classification, cell_label))

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    visualize_classifications(test_masks, test_classifications, "True classification")

    plt.subplot(1, 2, 2)
    visualize_classifications(
        test_masks,
        new_classifications,
        "Predicted classification Optimal parameters (he, L2:True, 8, relu)",
    )

    plt.tight_layout()
    plt.show()


def NN_classification():

    data_set = [2]
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "models", "he_relu_8_reg.pkl")
    model_path = os.path.normpath(model_path)
    model = load_neural_network(model_path)

    mask_file = os.path.join(
        script_dir,
        "..",
        "tests",
        "data",
        "classification",
        f"combined_mean_image{data_set[0]}_seg.npy",
    )
    mask_file = os.path.normpath(mask_file)

    test_classifications_file = os.path.join(
        script_dir,
        "..",
        "tests",
        "data",
        "classification",
        f"classifications{data_set[0]}.npy",
    )

    predict_and_visualize(model, data_set, mask_file, test_classifications_file)
    plt.show()


if __name__ == "__main__":
    NN_classification()
