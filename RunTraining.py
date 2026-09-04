import comet_ml
import os.path
import random
from typing import Literal
import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from torch.utils.data import DistributedSampler, Dataset, DataLoader
from tqdm import tqdm
import lightning
import hashlib
from model.MainModel import MainModel
import argparse
try:
    from xformers.ops import swiglu
    print(f"xformers SwiGLU is available.")
except:
    print(
        f"xformers SwiGLU is not available. Install xformers lib for better training performance."
    )
from RunInference import EncodingProteins
import orjson


class NewPDBDatasetModuleWithArbitraryBatchSize(pl.LightningDataModule):
    def __init__(
            self,
            trainingCSVs: tuple[str, str] | None,
            positiveBatchSizeForTraining: int,
            negativeBatchSizeForTraining: int,
            proteinEncodingPath: str,
            isESM: bool,
            workerNumber: int = 16,
            seed: int = 42,
            current_epoch = None,
    ) -> None:
        """

        """
        super().__init__()
        self.current_epoch = 0 if current_epoch is None else current_epoch
        self.trainCSVs = trainingCSVs
        # self.validationCSVs = validationCSVs
        self.positiveBatchSizeForTraining = positiveBatchSizeForTraining
        self.negativeBatchSizeForTraining = negativeBatchSizeForTraining
        self.workerNumber = workerNumber
        self.seed = seed
        self.proteinEncodingPath = proteinEncodingPath
        self.isESM = isESM

    def GetMonomerFeature(self, esmcFile=None, esm3File=None, encoderOutputFile=None) -> tuple:
        features = []
        if self.isESM:
            embeddingC = torch.load(esmcFile, map_location='cpu')[0][1: -1][:]
            embedding3 = torch.load(esm3File, map_location='cpu')[0][1: -1][:]
            features.append(embeddingC)
            features.append(embedding3)
        else:
            embedding = torch.load(encoderOutputFile, map_location='cpu').squeeze()
            features.append(embedding)
        output = torch.hstack(features)
        return output.unsqueeze(0)

    def GetMonomerString(self, ppiDataset, index):
        monomerA = str(ppiDataset['proteinIDA'][index])
        monomerB = str(ppiDataset['proteinIDB'][index])
        entryID = str(ppiDataset['entryID'][index])
        return monomerA, monomerB, entryID

    def GetEmbeddingsCollateFunction(
            self, inputMonomerIndexes, ppiDataset, monomerDict, negativeBatchSize,
            potentialInteractionPairID, esmcPath, esm3Path, interfacesPath, clusterIDMap
    ):

        # Load positive samples
        positives = []
        interfaces = []
        entryIDs = []
        clusterIDs = []
        for index in inputMonomerIndexes:
            monomerA, monomerB, entryID = self.GetMonomerString(ppiDataset, index)
            positives.append(monomerA)
            positives.append(monomerB)
            interfacePathA, interfacePathB = interfacesPath[monomerA][monomerB]
            interfaceA = torch.load(interfacePathA, map_location='cpu')
            interfaceB = torch.load(interfacePathB, map_location='cpu')
            interfaces.append(interfaceA)
            interfaces.append(interfaceB)
            entryIDs.append(entryID)
            clusterIDs.append(clusterIDMap[monomerA])
            clusterIDs.append(clusterIDMap[monomerB])

        # Find negative
        negatives = []
        while len(negatives) < negativeBatchSize:
            targetMonomer = random.choice(list(monomerDict))
            negativeID = monomerDict[targetMonomer]
            clusterID = clusterIDMap[targetMonomer]
            continueFlag = False
            if negativeID in entryIDs:
                print(f"Same ID in:{entryIDs}")
                continueFlag = True
            if clusterID in clusterIDs:
                print(f"Same cluster in:{clusterIDs}")
                continueFlag = True
            for positiveClusterID in clusterIDs:
                if clusterID in potentialInteractionPairID[positiveClusterID]:
                    print(f"Potential PPI partner for input cluster {clusterID} and positive cluster {positiveClusterID}")
                    continueFlag = True
            if continueFlag:
                continue
            negatives.append(targetMonomer)

        # Load features
        negativeFeatures = []
        for monomer in negatives:
            feature = self.GetMonomerFeature(esmcPath[monomer], esm3Path[monomer], f"{self.proteinEncodingPath}/{monomer}")
            negativeFeatures.append(feature)

        positiveFeatures = []
        for monomer in positives:
            feature = self.GetMonomerFeature(esmcPath[monomer], esm3Path[monomer], f"{self.proteinEncodingPath}/{monomer}")
            positiveFeatures.append(feature)

        return (positiveFeatures, negativeFeatures, interfaces), (positives, negatives)

    def GetPrecomputedResources(self, ppiDataFrame, resourceDataFrame):
        # used for create batches
        print(f"Preparing resources for dataloader...")
        monomers = {}
        potentialInteractionPairID = {}
        clusterIDs = {}
        esm3Path = {}
        esmcPath = {}
        interfaces = {}

        print(f"Reading resource dataframe...")
        for index in tqdm(range(len(resourceDataFrame))):
            # esm(3/c)EmbeddingFile
            esm3EmbeddingFile = resourceDataFrame['esm3EmbeddingFile'][index]
            esmcEmbeddingFile = resourceDataFrame['esmcEmbeddingFile'][index]
            proteinID = resourceDataFrame['proteinID'][index]
            if proteinID in esm3Path:
                if esm3Path[proteinID] != esm3EmbeddingFile:
                    print(f"Multiple ESM3 embeddings found for protein {proteinID}, use {esm3EmbeddingFile}")
            if proteinID in esmcPath:
                if esmcPath[proteinID] != esmcEmbeddingFile:
                    print(f"Multiple ESMC embeddings found for protein {proteinID}, use {esmcEmbeddingFile}")
            if not os.path.exists(esm3EmbeddingFile) or not os.path.exists(esmcEmbeddingFile):
                raise FileNotFoundError(f"Embedding files not found: {esm3EmbeddingFile} or {esmcEmbeddingFile}")
            esm3Path.update({proteinID: esm3EmbeddingFile})
            esmcPath.update({proteinID: esmcEmbeddingFile})

        print(f"Reading ppi dataframe...")
        for index in tqdm(range(len(ppiDataFrame))):
            # Get cluster ID & protein ID & entry ID
            monomerA = ppiDataFrame['proteinIDA'][index]
            monomerB = ppiDataFrame['proteinIDB'][index]
            clusterIDA = ppiDataFrame['clusterIDA'][index]
            clusterIDB = ppiDataFrame['clusterIDB'][index]
            entryID = ppiDataFrame['entryID'][index]
            if clusterIDA not in potentialInteractionPairID:
                potentialInteractionPairID.update({clusterIDA: []})
            if clusterIDB not in potentialInteractionPairID:
                potentialInteractionPairID.update({clusterIDB: []})
            potentialInteractionPairID[clusterIDA].append(clusterIDB)
            potentialInteractionPairID[clusterIDB].append(clusterIDA)

            if monomerB in monomers:
                if entryID != monomers[monomerB]:
                    print(f"Multiple entry ID found for protein {monomerB}, use {entryID}")
            monomers.update({monomerB: entryID})
            if monomerA in monomers:
                if entryID != monomers[monomerA]:
                    print(f"Multiple entry ID found for protein {monomerA}, use {entryID}")
            monomers.update({monomerA: entryID})

            if monomerB in clusterIDs:
                if clusterIDB != clusterIDs[monomerB]:
                    print(f"Multiple cluster ID found for protein {monomerB}, use {clusterIDB}")
            clusterIDs.update({monomerB: clusterIDB})
            if monomerA in clusterIDs:
                if clusterIDA != clusterIDs[monomerA]:
                    print(f"Multiple cluster ID found for protein {monomerA}, use {clusterIDA}")
            clusterIDs.update({monomerA: clusterIDA})

            if monomerA not in esm3Path:
                raise KeyError(f"Missing protein ID: {monomerA}, not found in embedding CSV")
            if monomerB not in esm3Path:
                raise KeyError(f"Missing protein ID: {monomerB}, not found in embedding CSV")
            # interface
            interfacesPathA = ppiDataFrame['interfaceA'][index]
            interfacesPathB = ppiDataFrame['interfaceB'][index]
            packFiles = (interfacesPathA, interfacesPathB)
            if not os.path.exists(interfacesPathA) or not os.path.exists(interfacesPathB):
                raise FileNotFoundError(f"Interface files not found: {interfacesPathA} or {interfacesPathB}")
            if monomerA not in interfaces:
                interfaces.update({monomerA: {monomerB: packFiles}})
                continue
            elif monomerB in interfaces[monomerA]:
                if packFiles != interfaces[monomerA][monomerB]:
                    print(f"Multiple interface files found for pair: {monomerA} and {monomerB}, use {packFiles}")
            interfaces[monomerA].update({monomerB: packFiles})
        return monomers, potentialInteractionPairID, esmcPath, esm3Path, interfaces, clusterIDs

    def GetDataLoader(
            self, ppiCSVFile, embeddingCSVFile, batchSizeForNegatives, batchSizeForPositives,
    ):
        # Load CSVs
        ppiDataFrame = pd.read_csv(ppiCSVFile, na_filter=False)
        resourceDataFrame = pd.read_csv(embeddingCSVFile, na_filter=False)
        # Reading files & sanity check
        set = PDBDatasetWithArbitraryBatchSize(ppiDataFrame)
        resoucesPack = self.GetPrecomputedResources(ppiDataFrame, resourceDataFrame)
        monomerDict, potentialInteractionPairID, esmcPath, esm3Path, interfaces, clusterIDs = resoucesPack
        # Collate function for negative mining
        collateFunction = lambda x: self.GetEmbeddingsCollateFunction(
            x, ppiDataFrame, monomerDict, batchSizeForNegatives, potentialInteractionPairID,
            esmcPath, esm3Path, interfaces, clusterIDs
        )
        distributedSampler = DistributedSampler(dataset=set, num_replicas=1, rank=0, seed=self.seed)
        # Data loader
        loader = DataLoader(
            dataset=set,
            sampler=distributedSampler,
            batch_size=batchSizeForPositives,
            num_workers=self.workerNumber,
            pin_memory=True,
            collate_fn=collateFunction,
            prefetch_factor=4,
            persistent_workers=True,
        )
        return loader

    def train_dataloader(self):
        if self.trainCSVs:
            ppiCSVFile, embeddingCSVFile = self.trainCSVs
            return self.GetDataLoader(
                ppiCSVFile, embeddingCSVFile, self.negativeBatchSizeForTraining, self.positiveBatchSizeForTraining
            )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None


class PDBDatasetWithArbitraryBatchSize(Dataset):
    def __init__(self, ppiDataFrame) -> None:
        super().__init__()
        self.ppiDataFrame = ppiDataFrame

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.ppiDataFrame)

def GetTrainingModel(
        trainingType: Literal["BinaryClassification", "ContrastiveHead"], logger,
        maxEpoch: int=15, annealingSteps: int=100000, warmupSteps: int=3000, checkpointPath: str = None
):

    hyperParams = {
        "name": "ppi",

        # conf mat for training
        "log_confusion_matrix_threshold": [0.7, 0.8, 0.9],
        "log_topk_arrange": [50, 100, 200],

        "log_confusion_matrix_cosine_threshold": [0.5, 0.6, 0.7, 0.9],
        "log_topk_cosine": [100, 200, 300, 400, 500, 600, 1000],

        # Block enc or dec
        "use_emb_only": False,
        "use_dec_only": False,

        # loss contribution
        "interface_factor": 0.0,
        "interaction_factor": 0.0,
        "contrastive_factor": 0.0,

        # Whether training use >12A padded interface as negative pair interfaces or not
        "padding_interface": False,

        "optimizer_config": {
            "optimizer": "AdamW",
            # loss的比例
            "learning_rate_adam": 1.6e-5,
            "learning_rate": 1.6e-5,
            "weight_decay": 0.05,
            "weight_decay_adam": 0.05,
            "min_lr": 4e-7,
            # epoch轮次
            "epochs": maxEpoch,
            # scheduler
            "warmup_steps": warmupSteps,
            "total_steps": annealingSteps,
        },

        "model": {
            'dropout': 0.1,
            # SigLIP or InfoNCE
            'loss_type': 'SigLIP',
            'loss_temperature': 1.0,
            'enc': {
                'conv_num': 3,
                'input_dim': 8704,  # 8704, # 1152+2560
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
    }
    if trainingType == "BinaryClassification":
        hyperParams['interface_factor'] = 1.0
        hyperParams['interaction_factor'] = 1.0
    elif trainingType == "ContrastiveHead":
        hyperParams['contrastive_factor'] = 1.0
    else:
        raise ValueError(f"Unknown training type: {trainingType}")
    model = MainModel(hyperParams)
    if checkpointPath is not None:
        print(f"Loading checkpoint from {checkpointPath}")
        pretrainedSupervisedModel = torch.load(checkpointPath, map_location=torch.device('cpu'))
        model.load_state_dict(pretrainedSupervisedModel['state_dict'], strict=True)
    logger.log_hyperparams(hyperParams)
    return model


def main():
    parser = argparse.ArgumentParser(description=f"Training the PPI Prediction Model")
    parser.add_argument(
        "--cometAPIKey",
        required=True,
        type=str,
        help="The API keys for Comet platform, where you can monitor your training progress",
    )
    parser.add_argument(
        '--cometExperimentName',
        default='PPI-Prediction',
        type=str,
        metavar='PPI-Prediction',
        help="Specify the experiment name showed on your Comet dashboard. "
             "and the model checkpoint will be saved at checkpoint/your experiment Name",
    )
    parser.add_argument(
        "--inputPPICSV",
        required=True,
        metavar="/path/to/PPIDataset.csv",
        type=str,
        help="Specify the CSV file path, which contains your PPI dataset",
    )
    parser.add_argument(
        "--inputEmbeddingCSV",
        metavar="/path/to/EmbeddingData.csv",
        required=True,
        type=str,
        help="Specify the CSV file path, which contains your embedding file directories",
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["BinaryClassification", "ContrastiveHead"],
        help="Specify the mode for training, you may train the classifier or finetune the contrastive head",
    )
    parser.add_argument(
        "--modelCheckpointPath",
        required=False,
        type=str,
        metavar="/path/to/modelCheckpoint.ckpt",
        help="Specify your model checkpoint path, if you would like to finetune the contrastive head or "
             "continue finetuning on other dataset for any trained models.",
    )
    parser.add_argument(
        '--encodingPath',
        metavar="/path/to/encoding",
        required=False,
        type=str,
        help="Specify the protein encoding file path, will be required by ContrastiveHead mode to "
             "restore its encoder's output.",
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
        default='cuda:0',
        type=str,
        metavar='cuda:N',
        help="Specify the device for training, we strongly recommend you specify a cuda device with VRAM>40G and "
             "install xformer lib which is compatible with your device. ",
    )
    parser.add_argument(
        '--batchSizeForPositiveSamples',
        default=2,
        type=int,
        metavar='2',
        help="Specify the positive PPI sample number in each batch.",
    )
    parser.add_argument(
        '--batchSizeForNegativeSamples',
        default=3,
        type=int,
        metavar='3',
        help="Specify the negative protein partner number in each batch. Remember that the positive-negative ratio "
             "is 1:2 * batchSizeForNegativeSamples, which default is 1:6",
    )
    parser.add_argument(
        '--cosineAnnealingStep',
        default=100000,
        type=int,
        metavar='100000',
        help="Specify the annealing step, we recommend you manually set it to (your dataset size * 10) "
             "for training classifier from scratch or (your dataset size * 5) for finetuning the contrastive head."
             "You may also tune it for the better performance on your own dataset.",
    )
    parser.add_argument(
        '--maxEpochs',
        default=10,
        type=int,
        metavar='10',
        help="Specify the maximum number of training epochs. ",
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        metavar='42',
        help="Specify the random seed. ",
    )


    cometAPIKey = parser.parse_args().cometAPIKey
    inputPPICSV = parser.parse_args().inputPPICSV
    inputEmbeddingCSV = parser.parse_args().inputEmbeddingCSV
    mode = parser.parse_args().mode
    modelCheckpointPath = parser.parse_args().modelCheckpointPath
    encodingPath = parser.parse_args().encodingPath
    dataloaderProcessNumber = parser.parse_args().dataloaderProcessNumber
    device = parser.parse_args().device
    batchSizeForPositiveSamples = parser.parse_args().batchSizeForPositiveSamples
    batchSizeForNegativeSamples = parser.parse_args().batchSizeForNegativeSamples
    cosineAnnealingStep = parser.parse_args().cosineAnnealingStep
    maxEpochs = parser.parse_args().maxEpochs
    cometExperimentName = parser.parse_args().cometExperimentName
    seed = parser.parse_args().seed

    lightning.seed_everything(seed, True)
    torch.set_float32_matmul_precision('medium')
    logger = CometLogger(
        save_dir="cometLogs",
        api_key=cometAPIKey,
        project_name="PPI-Project",
        experiment_name=cometExperimentName,
        mode='create'
    )

    datamodule = NewPDBDatasetModuleWithArbitraryBatchSize(
        trainingCSVs=(inputPPICSV, inputEmbeddingCSV),
        positiveBatchSizeForTraining=batchSizeForPositiveSamples,
        negativeBatchSizeForTraining=batchSizeForNegativeSamples,
        proteinEncodingPath=encodingPath,
        workerNumber=dataloaderProcessNumber,
        seed=seed,
        isESM=(mode == "BinaryClassification"),
    )
    os.makedirs(f"./checkpoints/{cometExperimentName}", exist_ok=True)
    checkpointCallback = ModelCheckpoint(
        dirpath=f"./checkpoints/{cometExperimentName}",
        filename="inter-model {epoch:d}",
        every_n_epochs=1,
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )

    model = GetTrainingModel(
        trainingType=mode,
        logger=logger,
        maxEpoch=maxEpochs,
        annealingSteps=cosineAnnealingStep,
        checkpointPath=modelCheckpointPath,
    )

    if 'cuda:' not in device:
        raise ValueError(f"Invalid device: {device}")
    else:
        deviceIDString: str = device.split(':')[-1]
    if not deviceIDString.isdigit():
        raise ValueError(f"Invalid device: {device}")
    deviceID = int(deviceIDString)

    if mode == 'ContrastiveHead':
        model.SetTrainingMode('finetuneEmbedder')
        print(f"Checking MD5...")
        with open(modelCheckpointPath, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()

        generateOutputFlag = True
        if os.path.exists(f"{encodingPath}/ModelCheckpointMD5.json"):
            with open(f"{encodingPath}/ModelCheckpointMD5.json", 'rb') as f:
                md5Old = orjson.loads(f.read())[0]
            if md5 != md5Old:
                print(f"MD5 failed, cache was generated by other model checkpoint.")
            else:
                print(f"MD5 Passed, use cached encoder output")
                generateOutputFlag = False
        if generateOutputFlag:
            print(f"Generating protein encoder embeddings")
            os.makedirs(encodingPath, exist_ok=True)
            EncodingProteins(encodingPath, inputEmbeddingCSV, model, dataloaderProcessNumber, deviceID)
            with open(f"{encodingPath}/ModelCheckpointMD5.json", 'wb') as f:
                f.write(orjson.dumps([md5]))

    trainer = pl.Trainer(
        devices=[deviceID],
        accelerator="gpu",
        precision='bf16-mixed',
        max_epochs=maxEpochs,
        min_epochs=maxEpochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[checkpointCallback],
        num_sanity_val_steps=0,
        deterministic="warn",
        limit_val_batches=0.0
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    main()

