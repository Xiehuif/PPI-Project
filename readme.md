# Interface-aware Binary Direct PPI Predictor
![Framework](./framework.bmp)

## Usage & License Restrictions
You are permitted to download and run this repository for personal use. However, incorporating, embedding, or redistributing any part of this code, documentation, or model weights into other projects, software, or libraries is strictly prohibited without prior written permission.

The manuscript for this work is currently in draft. Citation details and license granted will be updated upon publication.

All rights reserved.

## I. Installation & Requirement
### 1. Create conda env
Run the following command to create a conda environment for this project
```commandline
conda create --name PPIVal python=3.10
conda activate PPIVal
```

### 2. Install PyTorch
Follow the instructions of [the installation of different versions of PyTorch](https://pytorch.org/get-started/previous-versions/)
to install PyTorch compatible with your device.

We have tested our code on **RTX 5880 ADA & RTX A6000 with CUDA 12.4**:
```commandline
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install other Python packages
```commandline
pip install esm lightning comet_ml biopython pandas orjson pytorch_optimizer seaborn wget httpx
```
### 4. Install xformers (Optional, recommended for CUDA devices)
The xformers library improves performance on CUDA devices by providing a fused SwiGLU implementation, which reduces VRAM overhead and accelerates processing speed.
It is fine to run inference without xformers, and CPU-only users do not need to install it.
However, if you plan to train or fine-tune your own model, or run inference on large-scale datasets using GPU, we highly recommend installing xformers.

Refer to [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus) and [xformers
 repo.](https://github.com/facebookresearch/xformers) to install the xformers version compatible with your device and PyTorch version.

For **PyTorch 2.6.0 on RTX 5880 ADA & RTX A6000**, we install xformers using the following commands:
```commandline
conda install -c conda-forge gcc gxx
TORCH_CUDA_ARCH_LIST="8.6;8.9" pip install xformers==0.0.29.post3 --no-build-isolation
```

### 5. Install Modeller (Optional, required for extracting interface from CIF files in PDB database)
We demonstrate our procedure for extracting interaction interfaces from a CIF file in the script 
`RunInterfaceExtraction.py`, which requires Modeller for sequence alignment.

If you would like to reproduce our procedure to extract interaction interfaces from any CIF file 
(e.g., for training or validation), you need to install Modeller by following 
[their official instruction](https://salilab.org/modeller/10.8/release.html#anaconda).

## II. Run Inference
To run inference, you need to [download the model weight](https://pan.baidu.com/s/1AY1fm8eYSWR6ZecCnh7LQg?pwd=4qnn).
### 0. Preparing Inputs
We provided input files to demonstrate our inference pipeline. They contain recent PDB records released 
after August 2026, ensuring they were excluded from both our training and validation datasets.
```
./examples/Example.csv 
./examples/ExamplePair.csv
```

---
`Example.csv` is a **protein sequence description file**, which must contain at least two columns with header **proteinID** 
and **sequence**, corresponding to your protein name/code/id and its sequence in single letter.

---
`ExamplePair.csv` is a **protein pair description file**, which must contain at least two columns with header 
**proteinIDA** and **proteinIDB**, corresponding to the PPI you would like to investigate. **proteinIDA** & 
**proteinIDB** must be aligned with the **proteinID** in your **protein sequence description file**.
---

This section demonstrates how to extract the protein vectors for protein chains in `Example.csv`, 
and extract the interaction interfaces & probabilities for the pair in `ExamplePair.csv`


### 1.  Generating ESM 3/ESM C Embeddings
 
Follow the instruction on the [BioHub developer console](https://biohub.ai/developer-console/api-keys) to 
create an account and generate an API-Key for your service. Then, run:
```commandline
python RunEmbedding.py --apiKey <your_api_key> --inputSequences ./examples/Example.csv --output ./output
```
Replace the `<your_api_key>` with your API Key acquired from BioHub. This stage helps you to construct ESM 3/ESM C 
embeddings, which would be stored in `./output/ESM3` & `./output/ESMC`, 
and an **embedding description file** which records the embedding path of each protein. 

### 2. Encoding Proteins
Run the following command to encode proteins from ESM 3/ESM C embeddings.
```commandline
python RunInference.py --mode Encoding --inputCSV ./output/MonomerEmbeddingFiles.csv --output ./output/encodingPath --modelCheckpointPath <path_to_your_model_parameter> --device cuda:0
```
Replace the `<path_to_your_model_parameter>` with the path of model weight file. 
This will make directory `./output/encodingPath` as the output directory of **protein encoder** (--encodingPath).

### 3. Extract Protein Vectors from Encoder Outputs
Run the following command to extract protein vectors using **Hidden Space Projector**.
```commandline
python RunInference.py --mode Embedding --inputCSV ./output/MonomerEmbeddingFiles.csv --output ./output --modelCheckpointPath <path_to_your_model_parameter> --encodingPath ./output/encodingPath/ --device cuda:0
```
It will generate a json file which contains a dictionary with format `{"proteinID": proteinVector}` at:
```
./output/ProteinVectors.json
```

### 4. Extract PPI Probabilities & Interface
Run the following command to extract probabilities & interface for the protein pair in **protein pair description file**
:
```commandline
python RunInference.py --mode BinaryClassification --inputCSV ./examples/ExamplePair.csv --output ./output --modelCheckpointPath <path_to_your_model_parameter> --encodingPath ./output/encodingPath/ --device cuda:0
python RunInference.py --mode Interface --inputCSV ./examples/ExamplePairPositives.csv --output ./output --modelCheckpointPath <path_to_your_model_parameter> --encodingPath ./output/encodingPath/ --device cuda:0
```
It will generate two JSON files, which contain a dictionary with format `{"proteinIDA:proteinIDB": logits}` 
corresponding to interaction probabilities & interfaces at:
```
./output/BinaryClassificationLogits.json
./output/InterfacesLogits.json
```

### 5. Result Analysis
we provide `ExampleOutputAnalysis.py` & `ExampleInterfaceAnalysis.py` script to demonstrate how we analyze model outputs. 

un the following commands to display the results of our examples in a readable format:
```commandline
python ExampleOutputAnalysis.py
python ExampleInterfaceAnalysis.py
```

#### For protein vectors:
```python
import orjson
import os
import numpy as np

def GetSimilarityScore(proteinIDA, proteinIDB, fileName):
    with open(fileName, 'rb') as f:
        data = orjson.loads(f.read())
    vectorA = np.array(data[proteinIDA][0])
    vectorB = np.array(data[proteinIDB][0])
    vectorA = vectorA / np.linalg.norm(vectorA)
    vectorB = vectorB / np.linalg.norm(vectorB)
    similarityScore = np.dot(vectorA, vectorB)
    print(f"Similarity for pair {proteinIDA}:{proteinIDB} is {similarityScore}")
    return similarityScore


if __name__ == '__main__':
    # Some actions...
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
    # Some actions...
```

#### For classification logits
```python
import orjson
import os
import numpy as np

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


if __name__ == '__main__':
    # Some actions...
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
    # Some actions...
```

#### For interface
```python
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
```

## III. Extracting Interface from CIF Files
With Modeller installed properly, you may use `RunInterfaceExtraction.py` to extract interface of any PPI pair in any 
CIF files.

### 0. Preparing Inputs
Here, We demonstrate how to extract the interface of chain [E & H in PDB entry 43NO](https://www.rcsb.org/structure/43NO).
 Following files will be necessary.
```
./examples/exampleCIF/43NO.cif 
./examples/ExampleInterface.csv
```
---
`43NO.cif` is CIF file of [biological assembly 1](https://files.rcsb.org/download/43NO-assembly1.cif.gz) in PDB entry 43NO. 

---
`ExampleInterface.csv` is a **protein pair interface description file**, which contains 7 columns:
```
proteinIDA: Name of your chain A.
proteinIDB: Name of your chain B.
sequenceA: sequence of your chain A, you can read it from PDB FASTA file.
sequenceB: sequence of your chain B.
chainIDA: Author ID of your chain A. 
chainIDB: Author ID of your chain B.
structureFile: Path to your CIF file.
```

---

### 1. Extracting Interface
Run the command:
```commandline
python RunInterfaceExtraction.py --inputCSV ./examples/ExampleInterface.csv --output ./output
```
Three folders will be created after the run:
```
./output/alignments: FASTA Sequence & CIF modelled sequence alignment.
./output/distanceMaps: Distance map between residues in two different chains.
./output/interfaces: Interface file.
```

### 2. Validation 
See `ExampleInterfaceAnalysis.py`.
```python
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

```

## IV. Run Training (Customizing for your own model)
By generating the protein ESM 3/C embedding and extracting interface from your dataset, you may train your own model
 checkpoint with our script `RunTraining.py`, which also demonstrates our training protocol.

Please make sure you have the two CSV files prepared:

---
`MonomerEmbeddingFiles.csv`: File Generated by `RunEmbeddings.py`, clarifying the embedding paths of all proteins. 

---

`PPI.csv`: File with 7 columns, clarifying the interface file path, all the protein-protein interaction ,the cluster 
id of all protein monomers and their PDB entry ID:
```
proteinIDA: Name of your chain A, must be aligned with MonomerEmbeddingFiles.csv.
proteinIDB: Name of your chain B.
clusterIDA: sequence of your chain A, you can read it from PDB FASTA file.
clusterIDB: sequence of your chain B.
interfaceA: Path to your chain A interface file. 
interfaceB: Path to your chain B interface file.
entryID: PDB Entry ID of this PPI.
```

---
We prepared an example `./examples/TrainingSet_Demo.csv` & `./examples/TrainingSet_Demo_Monomer.csv` which correspond to the dataset we use to train our public checkpoint.

Register a Comet account, get your own Comet API Key(<your_comet_api_key>) to monitor your training process.

For **Stage I (Classifier Pretraining)** run:
```commandline
python RunTraining.py --cometAPIKey <your_comet_api_key> --cometExperimentName StageI --inputPPICSV ./examples/TrainingSet_Demo.csv --inputEmbeddingCSV ./examples/TrainingSet_Demo_Monomer.csv --mode BinaryClassification
```

For **Stage II (Contrastive Head Finetuning)** run:
```commandline
python RunTraining.py --cometAPIKey <your_comet_api_key> --cometExperimentName StageII --inputPPICSV ./examples/TrainingSet_Demo.csv --inputEmbeddingCSV ./examples/TrainingSet_Demo_Monomer.csv --mode ContrastiveHead --encodingPath ./encodingTemp --batchSizeForNegativeSamples 32 --cosineAnnealingStep 50000 --maxEpochs 5 --device cuda:0 --modelCheckpointPath <your_pretrained_classifier_model>
```

We have our training demonstrated publicly on our [Comet project website](https://www.comet.com/xiehuif/ppi-project/view/new/panels).




