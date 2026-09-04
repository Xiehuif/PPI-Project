import os
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import seaborn as sns
import torch

def GetInterfacePrediction(data, pair, chainIndex):
    interface = data[pair][chainIndex][0]
    def softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    interface = softmax(np.array(interface), axis=-1)
    return interface

def ShowDoubleMap(data1, data2, title1, title2, pathToSave=None):
    def get_columns(data):
        if data.shape[1] == 4:
            return ["<4A", "4-8A", "8-12A", ">12A"]
        elif data.shape[1] == 5:
            return ["Unmodelled", "<4A", "4-8A", "8-12A", ">12A"]
        else:
            return [f"Col_{i}" for i in range(data.shape[1])]

    df1_T = pd.DataFrame(data1, columns=get_columns(data1)).T
    df2_T = pd.DataFrame(data2, columns=get_columns(data2)).T
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 6), sharex=True, gridspec_kw={"hspace": 0.3}
    )
    sns.heatmap(
        df1_T,
        ax=ax1,
        cmap="coolwarm",
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"label": "Bin Logits", "pad": 0.02},
    )
    ax1.set_title(title1, fontsize=12, pad=8)
    ax1.tick_params(axis="y", rotation=0, labelsize=10)
    sns.heatmap(
        df2_T,
        ax=ax2,
        cmap="coolwarm",
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"label": "Prediction Probability", "pad": 0.02},
    )
    ax2.set_title(title2, fontsize=12, pad=8)
    ax2.set_xlabel("Residues", fontsize=10)
    ax2.tick_params(axis="y", rotation=0, labelsize=10)
    plt.tight_layout()
    if pathToSave is not None:
        plt.savefig(pathToSave, dpi=600)
    else:
        plt.show()

# Your prediction output
interfacePath = './output/InterfacesLogits.json'
if os.path.exists(interfacePath):
    with open(interfacePath) as json_file:
        data = orjson.loads(json_file.read())
    predictionA = GetInterfacePrediction(data, "10DW-A:10DW-C", 1)
    predictionB = GetInterfacePrediction(data, "10DW-C:10DW-B", 0)
    ShowDoubleMap(np.array(predictionA), np.array(predictionB), '10DW-C-AC-C', '10DW-C-CB-C',
                  './output/InferenceDemo.png')
else:
    print(f"File Not Found: {interfacePath}")

# Your ground truth output
interfaceGroundTruthA = f"./output/interfaces/10DW-C-AC-C"
interfaceGroundTruthB = f"./output/interfaces/10DW-C-CB-C"
if os.path.exists(interfaceGroundTruthA) and os.path.exists(interfaceGroundTruthB):
    groundTruthA = torch.load(f"./output/interfaces/10DW-C-AC-C")
    groundTruthB = torch.load(f"./output/interfaces/10DW-C-CB-C")
    ShowDoubleMap(np.array(groundTruthA), np.array(groundTruthB), '10DW-C-AC-C', '10DW-C-CB-C',
                  './output/GroundTruthDemo.png')
else:
    print(f"File Not Found: {interfaceGroundTruthA} & {interfaceGroundTruthB}")
