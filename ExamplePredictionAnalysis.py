import orjson
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def GetSimilarityScore(proteinIDA, proteinIDB, fileName):
    # Open & read the file
    with open(fileName, 'rb') as f:
        data = orjson.loads(f.read())
    vectorA = np.array(data[proteinIDA][0])
    vectorB = np.array(data[proteinIDB][0])
    # L2 Norm
    vectorA = vectorA / np.linalg.norm(vectorA)
    vectorB = vectorB / np.linalg.norm(vectorB)
    # Dot product
    similarityScore = np.dot(vectorA, vectorB)
    # Output
    print(f"Similarity for pair {proteinIDA}:{proteinIDB} is {similarityScore}")
    return similarityScore

def GetInteractionLogits(proteinIDA, proteinIDB, fileName):
    # Open & read the file
    with open(fileName, 'rb') as f:
        data = orjson.loads(f.read())
    if f"{proteinIDA}:{proteinIDB}" in data:
        prediction = data[f"{proteinIDA}:{proteinIDB}"]
    elif f"{proteinIDB}:{proteinIDA}" in data:
        prediction = data[f"{proteinIDB}:{proteinIDA}"]
    else:
        print(f"No protein pair found for {proteinIDA} & {proteinIDB}")
        return
    # We use min logit as prediction
    prediction = min(prediction[0][0][0], prediction[1][0][0])
    # Output
    print(f"Interaction scores for pair {proteinIDA}:{proteinIDB} is {prediction}")
    return prediction

def GetInterfacePrediction(data, pair, chainIndex):
    interface = data[pair][chainIndex][0]
    def softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    interface = softmax(np.array(interface), axis=-1)
    return interface

def ShowDoubleMap(data1, data2, title1, title2):
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
    plt.show()

if __name__ == '__main__':
    outputVectorDatabase = "./output/ProteinVectors.json"
    if os.path.exists(outputVectorDatabase):
        print(f"For positive pairs:")
        GetSimilarityScore('43NO-ABEF', '43NO-CDGH', outputVectorDatabase)
        GetSimilarityScore('9RON-A', '9RON-B', outputVectorDatabase)
        GetSimilarityScore('9RON-A', '9RON-C', outputVectorDatabase)
        GetSimilarityScore('37HF-AC', '37HF-BD', outputVectorDatabase)
        GetSimilarityScore('22QT-ABC', '22QT-DE', outputVectorDatabase)
        print(f"For random pairs, we can consider they are the negatives:")
        GetSimilarityScore('22QT-ABC', '37HF-BD', outputVectorDatabase)
        GetSimilarityScore('37HF-AC', '9RON-A', outputVectorDatabase)
        GetSimilarityScore('22QT-DE', '43NO-CDGH', outputVectorDatabase)
        GetSimilarityScore('43NO-ABEF', '37HF-BD', outputVectorDatabase)
        GetSimilarityScore('9RON-B', '9RON-C', outputVectorDatabase)
    else:
        print(f"You have not run the inference example yet.")

    outputInteractionLogits = "./output/BinaryClassificationLogits.json"
    if os.path.exists(outputInteractionLogits):
        print(f"For positive pairs:")
        GetInteractionLogits('43NO-ABEF', '43NO-CDGH', outputInteractionLogits)
        GetInteractionLogits('9RON-A', '9RON-B', outputInteractionLogits)
        GetInteractionLogits('9RON-A', '9RON-C', outputInteractionLogits)
        GetInteractionLogits('37HF-AC', '37HF-BD', outputInteractionLogits)
        GetInteractionLogits('22QT-ABC', '22QT-DE', outputInteractionLogits)
        print(f"For random pairs, we can consider they are the negatives:")
        GetInteractionLogits('22QT-ABC', '37HF-BD', outputInteractionLogits)
        GetInteractionLogits('37HF-AC', '9RON-A', outputInteractionLogits)
        GetInteractionLogits('22QT-DE', '43NO-CDGH', outputInteractionLogits)
        GetInteractionLogits('43NO-ABEF', '37HF-BD', outputInteractionLogits)
        GetInteractionLogits('9RON-B', '9RON-C', outputInteractionLogits)
    else:
        print(f"You have not run the inference example yet.")

    interfacePath = './output/InterfacesLogits.json'
    with open(interfacePath) as json_file:
        data = orjson.loads(json_file.read())
    samples = [
        "22QT-ABC:22QT-DE",
        "43NO-ABEF:43NO-CDGH",
        '9RON-A:9RON-B',
        '37HF-AC:37HF-BD',
        '9RON-A:9RON-C',
    ]
    for sample in samples:
        predictionA = GetInterfacePrediction(data, sample, 0)
        predictionB = GetInterfacePrediction(data, sample, 1)
        print(f"For pairs {sample} shape: {predictionA.shape} and {predictionB.shape}")
        ShowDoubleMap(np.array(predictionB), np.array(predictionA),
                      f'{sample}-1', f'{sample}-0')