import argparse
from model.InferenceWriters import UniprotProteomeEmbeddingWriter, ClassificationWriter, UniprotProteomeEncodingWriter
import lightning.pytorch as pl
import lightning
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
from model.MainModel import MainModel
try:
    from xformers.ops import swiglu
    print(f"xformers SwiGLU is available.")
except:
    print(
        f"xformers SwiGLU is not available. If you running on CPU, you may ignore this warning; "
        f"If you running on GPU, install xformers for better performance."
    )


class MonomerDataset(Dataset):

    def __init__(self, monomerCSV) -> None:
        print(f"Reading input monomer dataset...")
        self.monomerEmbeddingFiles = {}
        self.monomerLabels = []
        for index, row in pd.read_csv(monomerCSV, na_filter=False).iterrows():
            proteinID = row['proteinID']
            esm3File = row['esm3EmbeddingFile']
            esmcFile = row['esmcEmbeddingFile']
            files = [esmcFile, esm3File]
            if proteinID in self.monomerEmbeddingFiles:
                if not self.monomerEmbeddingFiles[proteinID] == files:
                    print(
                        f"WARNING: found protein ID {proteinID} for multiple times, "
                        f"last time get ESM files {self.monomerEmbeddingFiles[proteinID]},"
                        f"overwritten as {files}"
                    )
            if not os.path.exists(esm3File):
                raise FileNotFoundError(f"Can not found esm3 embedding:{esm3File}")
            if not os.path.exists(esmcFile):
                raise FileNotFoundError(f"Can not found esmc embedding:{esmcFile}")
            self.monomerEmbeddingFiles.update({proteinID: files})
            if proteinID not in self.monomerLabels:
                self.monomerLabels.append(proteinID)

    def GetMonomerFeature(self, monomer: str) -> tuple:
        features = []
        esmcFile, esm3File = self.monomerEmbeddingFiles[monomer]
        embeddingC = torch.load(esmcFile, map_location='cpu')[0][1: -1][:]
        embedding3 = torch.load(esm3File, map_location='cpu')[0][1: -1][:]
        features.append(embeddingC)
        features.append(embedding3)
        return torch.hstack(features)

    def __getitem__(self, index):
        targetMonomer = self.monomerLabels[index]
        # positive: nodes, edges, edge_index
        feature = self.GetMonomerFeature(targetMonomer)
        return feature, targetMonomer

    def __len__(self):
        return len(self.monomerLabels)

class MonomerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        monomerCSVDir,
        num_workers: int = 16,
    ) -> None:

        super().__init__()
        self.csvDir = monomerCSVDir
        self.num_workers = num_workers

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        set = MonomerDataset(self.csvDir)
        loader = DataLoader(dataset=set, batch_size=1, num_workers=self.num_workers, pin_memory=True)
        return loader

class InteractionDataset(Dataset):
    def __init__(self, ppiCSV) -> None:
        super().__init__()
        self.ppiDF = pd.read_csv(ppiCSV, na_filter=False)

    def __getitem__(self, index):
        return self.ppiDF['proteinIDA'][index], self.ppiDF['proteinIDB'][index]

    def __len__(self):
        return len(self.ppiDF['proteinIDA'])

class InteractionDataModule(pl.LightningDataModule):
    def __init__(
            self,
            targetCSV,
            encoderOutputPath,
            useRAMCache = False,
            num_workers: int = 16,
            device='cpu',
        ) -> None:

        super().__init__()
        self.num_workers = num_workers
        if useRAMCache:
            self.num_workers = 1
        self.csvPath = targetCSV
        self.dataDir = encoderOutputPath
        self.device = device if device == 'cpu' else torch.device(f'cuda:{device[0]}')
        # use RAM cache for better IO performance
        # yet, a large part of RAM memory may be consumed, specific consumption depends on your dataset size
        self.cache = {}
        dataFrame = pd.read_csv(targetCSV, na_filter=False)
        uniprotIDAList = dataFrame['proteinIDA']
        uniprotIDBList = dataFrame['proteinIDB']
        fullUniprotID = []
        print(f"Reading Protein IDs...")
        for uniprotID in tqdm(uniprotIDAList):
            if uniprotID not in fullUniprotID:
                fullUniprotID.append(uniprotID)
        for uniprotID in tqdm(uniprotIDBList):
            if uniprotID not in fullUniprotID:
                fullUniprotID.append(uniprotID)
        if useRAMCache:
            print(f"Reading features...")
            for uniprotID in tqdm(fullUniprotID):
                self.GetMonomerFeature(uniprotID)
        else:
            print(f"Checking features...")
            for uniprotID in tqdm(fullUniprotID):
                fileDir = f"{self.dataDir}/{uniprotID}"
                if not os.path.exists(fileDir):
                    raise FileNotFoundError(f"Can not found encoder output file:{fileDir}")

    def GetMonomerFeature(self, monomer: str) -> tuple:
        if monomer in self.cache:
            return self.cache[monomer]
        features = []
        embeddingPath = f"{self.dataDir}/{monomer}"
        embedding = torch.load(embeddingPath, map_location='cpu').squeeze()
        features.append(embedding)
        returnValue = torch.hstack(features).unsqueeze(0)
        self.cache.update({monomer: returnValue})
        return returnValue

    def GetEmbeddingsCollateFunction(self, inputMonomers):
        monomerA, monomerB = inputMonomers[0]
        featureA = self.GetMonomerFeature(monomerA)
        featrueB = self.GetMonomerFeature(monomerB)
        return (featureA, featrueB), inputMonomers

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        if self.csvPath:
            set = InteractionDataset(self.csvPath)
            collateFunction = self.GetEmbeddingsCollateFunction
            loader = DataLoader(dataset=set, batch_size=1,
                                num_workers=self.num_workers, pin_memory=True, collate_fn=collateFunction)
            return loader

class EncodingDataset(Dataset):

    def __init__(self, inputEncodingDir) -> None:
        print(f"Reading all protein encoding output...")
        self.encoderOutputPath = inputEncodingDir
        self.proteinEncodingOutput = os.listdir(inputEncodingDir)

    def GetMonomerFeature(self, monomer: str) -> tuple:
        features = []
        embedding = torch.load(f"{self.encoderOutputPath}/{monomer}", map_location='cpu').squeeze()
        features.append(embedding)
        return torch.hstack(features)

    def __getitem__(self, index):
        targetMonomer = self.proteinEncodingOutput[index]
        # positive: nodes, edges, edge_index
        feature = self.GetMonomerFeature(targetMonomer)
        return feature, targetMonomer

    def __len__(self):
        return len(self.proteinEncodingOutput)

class EncodingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        encodingDir,
        num_workers: int = 16,
    ) -> None:

        super().__init__()
        self.encodingDir= encodingDir
        self.num_workers = num_workers

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        set = EncodingDataset(self.encodingDir)
        loader = DataLoader(dataset=set, batch_size=1, num_workers=self.num_workers, pin_memory=True)
        return loader

def LoadModel(modelCheckpointPath):
    lightning.seed_everything(42, True)
    torch.set_float32_matmul_precision('medium')
    hyper_params = {
        "name": "ppi",

        # conf mat for training
        "log_confusion_matrix_threshold": [0.7, 0.8, 0.9],
        "log_topk_arrange": [50, 100, 200],

        "log_confusion_matrix_cosine_threshold": [0.5, 0.6, 0.7, 0.9],
        "log_topk_cosine": [100, 200, 300, 400, 500, 600, 1000, 2000],

        # loss contribution
        "interface_factor": 0.0,
        "interaction_factor": 0.0,
        "contrastive_factor": 0.0,

        # Block enc or dec
        "use_emb_only": False,
        "use_dec_only": False,
        "optimizer_config": {
            "optimizer": "AdamW",
            # loss的比例
            "learning_rate_adam": 1.6e-5,
            "learning_rate": 1.6e-5,
            "weight_decay": 0.05,
            "weight_decay_adam": 0.05,
            "min_lr": 4e-7,
            # epoch轮次
            "epochs": 15,
            # scheduler
            "warmup_steps": 3000,
            "total_steps": 135000,
        },

        "model": {
            'dropout': 0.1,
            # SigLIP or InfoNCE
            'loss_type': 'SigLIP',
            'loss_temperature': 1.0,
            'enc': {
                'conv_num': 3,
                'input_dim': 8704,  # 8704 all 2560 C 6144 3
                'hidden_dim': 1920,
            },
            'emb': {
                'ffn_factor': 3,
                'head_num': 30,
                'layer_num': 3,
            },
            'dec': {
                'head_num': 30,
                'layer_num': 3,
                'arch': 'plain',
            },
        },
        "precision": 'bf16-mixed',
        "load_ckpt": None,
    }
    model = MainModel(hyper_params)
    pretrainedModel = torch.load(modelCheckpointPath, map_location=torch.device('cpu'))
    model.load_state_dict(pretrainedModel['state_dict'], strict=True)
    return model

def EncodingProteins(outputPath, inputCSV, model, numberOfWorkers=16, devices='cpu'):
    model.SetDataFormatToMonomerInput()
    model.InferenceEncoding()
    os.makedirs(outputPath, exist_ok=True)
    dataModule = MonomerDataModule(inputCSV, numberOfWorkers)
    outputWriter = UniprotProteomeEncodingWriter(outputPath, write_interval='batch_and_epoch', inputType='Monomer')
    trainer = pl.Trainer(
        devices=devices,
        accelerator="cpu" if devices=='cpu' else "gpu",
        precision="bf16-mixed",
        log_every_n_steps=1,
        callbacks=[outputWriter],
        num_sanity_val_steps=0,
        deterministic="warn",
    )
    trainer.predict(model=model, dataloaders=dataModule)

def EmbeddingProteins(inputEncodingPath, outputPath, model, numberOfWorkers=16, devices='cpu'):
    model.SetDataFormatToMonomerInput()
    model.InferenceEmbedding()
    dataModule = EncodingDataModule(inputEncodingPath, numberOfWorkers)
    outputWriter = UniprotProteomeEmbeddingWriter(
        outputPath, write_interval='batch_and_epoch', inputType='Monomer'
    )
    trainer = pl.Trainer(
        devices=devices,
        accelerator="cpu" if devices == 'cpu' else "gpu",
        precision="bf16-mixed",
        log_every_n_steps=1,
        callbacks=[outputWriter],
        num_sanity_val_steps=0,
        deterministic="warn",
    )
    trainer.predict(model=model, dataloaders=dataModule)

def ClassifyProteinPairs(
        inputEncodingPath, inputCSV, outputFile, model,
        numberOfWorkers=16, devices='cpu', useRAMCache=True
):
    model.InferenceClassificationScore()
    model.SetDataFormatToPairwiseInput()
    predictionWriter = ClassificationWriter(
        outputFile,
        'batch_and_epoch'
    )
    dataModule = InteractionDataModule(
        targetCSV=inputCSV,
        num_workers=numberOfWorkers,
        encoderOutputPath=inputEncodingPath,
        useRAMCache=useRAMCache,
        device=devices,
    )
    trainer = pl.Trainer(
        devices=devices,
        accelerator="cpu" if devices == 'cpu' else "gpu",
        precision="bf16-mixed",
        log_every_n_steps=1,
        callbacks=[predictionWriter],
        num_sanity_val_steps=0,
        deterministic="warn",
    )
    trainer.predict(model=model, dataloaders=dataModule)

def GetInterface(
        inputEncodingPath, inputCSV, outputFile, model,
        numberOfWorkers=16, devices='cpu', useRAMCache=True
):
    model.InferenceInterface()
    model.SetDataFormatToPairwiseInput()
    predictionWriter = ClassificationWriter(
        outputFile,
        'batch_and_epoch'
    )
    dataModule = InteractionDataModule(
        targetCSV=inputCSV,
        num_workers=numberOfWorkers,
        encoderOutputPath=inputEncodingPath,
        useRAMCache=useRAMCache,
        device=devices,
    )
    trainer = pl.Trainer(
        devices=devices,
        accelerator="cpu" if devices == 'cpu' else "gpu",
        precision="bf16-mixed",
        log_every_n_steps=1,
        callbacks=[predictionWriter],
        num_sanity_val_steps=0,
        deterministic="warn",
    )
    trainer.predict(model=model, dataloaders=dataModule)

def main():
    parser = argparse.ArgumentParser(description=f"Model inference entrance for PPI prediction")
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["BinaryClassification", "Encoding", "Embedding", "Interface"],
        help="Specify the mode for inference engine to run, see readme.md for details",
    )
    parser.add_argument(
        "--inputCSV",
        required=True,
        metavar="input.csv",
        type=str,
        help="Specify the input CSV file path, see readme.md for details",
    )
    parser.add_argument(
        "--output",
        metavar="/path/to/outputDir",
        required=True,
        type=str,
        help="Specify the output path, must be a directory. If the output directory does not exist, it will be created.",
    )
    parser.add_argument(
        '--modelCheckpointPath',
        metavar='/path/to/model.ckpt',
        required=True,
        type=str,
        help="Model parameter file path.",
    )
    parser.add_argument(
        '--encodingPath',
        metavar="/path/to/encoding",
        required=False,
        type=str,
        help="Specify the protein encoding file path, will be required by BinaryClassification mode and Embedding mode.",
    )
    parser.add_argument(
        '--dataloaderProcessNumber',
        default=16,
        type=int,
        metavar='16',
        help="Specify the dataloader process number, default is 16, which achieve a good performance-load balance "
             "for our in-house test. You should setting this argument to adapt to your machine specs."
             "If ram cache is enabled, this argument will be ignored for BinaryClassification mode.",
    )
    parser.add_argument(
        '--device',
        default='cpu',
        type=str,
        metavar='cpu or cuda:N',
        help="Specify the device for inference, though default is cpu, we recommend cuda device. If you have multiple GPUs, "
             "you should specify the GPU you would like to use by setting this argument to like cuda:N."
             "We now only support working on a single GPU, if you would like to use multiple GPUs, "
             "please manually split your dataset and run multiple processes for each of parts.",
    )
    parser.add_argument(
        '--enableRAMCache',
        default=False,
        type=bool,
        metavar='False',
        help="Specify whether RAM cache is enabled, default is False. "
             "It only affects the performance of BinaryClassification mode."
             "When this argument is true, pre-encoding results of every proteins will be loaded to RAM in advance. "
             "Which reduce the IO overhead between the disk and RAM, since the encoding result of a single protein may "
             "be used for several times in BinaryClassification mode. "
             "However, it may consume quite a part of memory (10-100GB for ~10^4 protein chains, "
             "depends on your data scale)."
             "While enabling cache may lead to a better performance, it may cause OOM error",
    )


    mode = parser.parse_args().mode
    if mode != 'Embedding':
        inputCSV = parser.parse_args().inputCSV
    outputPath = parser.parse_args().output
    modelCheckpointPath = parser.parse_args().modelCheckpointPath
    if mode == "BinaryClassification" or mode == "Embedding" or mode == "Interface":
        encodingPath = parser.parse_args().encodingPath
    else:
        encodingPath = None

    model = LoadModel(modelCheckpointPath)

    enableRAMCache = parser.parse_args().enableRAMCache
    numberOfWorkers = parser.parse_args().dataloaderProcessNumber
    device = parser.parse_args().device
    os.makedirs(outputPath, exist_ok=True)

    if device != 'cpu':
        devicesParse = device.split(':')
        if len(devicesParse) != 2 or devicesParse[0] != 'cuda':
            raise ValueError(f"Device is not supported: {device}")
        else:
            try:
                devicesParse = [int(devicesParse[1])]
            except:
                raise ValueError(f"Device is not supported: {device}")
        device = devicesParse

    if mode == "BinaryClassification":
        outputFile = f"{outputPath}/BinaryClassificationLogits.json"
        ClassifyProteinPairs(encodingPath, inputCSV, outputFile, model, numberOfWorkers, device, enableRAMCache)
    if mode == "Encoding":
        EncodingProteins(outputPath, inputCSV, model, numberOfWorkers, device)
    if mode == "Embedding":
        outputFile = f"{outputPath}/ProteinVectors.json"
        EmbeddingProteins(encodingPath, outputFile, model, numberOfWorkers, device)
    if mode == 'Interface':
        outputFile = f"{outputPath}/InterfacesLogits.json"
        GetInterface(encodingPath, inputCSV, outputFile, model, numberOfWorkers, device, enableRAMCache)


if __name__ == '__main__':
    main()
