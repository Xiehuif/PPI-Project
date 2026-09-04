from typing import Any, Literal

import torch,os,sys,comet_ml
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_optimizer import Muon
import seaborn as sns
from torch import Tensor, nn
from math import sqrt

from torch.nn import Embedding
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import model.Loss

sys.path.append(os.getcwd())

import torch,torch.nn
import torch.nn.functional as F

import lightning.pytorch as pl
import torchmetrics.classification as mc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

from model.ProteinEncoder import BertEncoder
from model.ProteinEmbedder import Embedder
from model.ProteinDecoder import SymmetricTransformerDecoder, PlainTransformerDecoder


class MainModel(pl.LightningModule):

    def on_train_epoch_start(self) -> None:
        print(f"Reset dataloader seed to epoch {self.current_epoch}")
        loader = self.trainer.train_dataloader
        loader.sampler.set_epoch(self.current_epoch)
        self.val_step = 0

    def __init__(self, hyperParameter):
        super().__init__()
        self.hyperParameter = hyperParameter

        matplotlib.use('agg')
        plt.switch_backend('agg')
        print(f"plt backend:{plt.get_backend()}")
        print(f"matplotlib backend:{matplotlib.get_backend()}")
        self.proteinEncoder = BertEncoder(
            encoderLayerNumber=hyperParameter['model']['enc']['conv_num'],
            inputNodeDimension=hyperParameter['model']['enc']['input_dim'],
            hiddenNodeDimension=hyperParameter['model']['enc']['hidden_dim'],
            ffnFactor=hyperParameter['model']['emb']['ffn_factor'],
            headNumber=hyperParameter['model']['emb']['head_num'],
            dropout=hyperParameter['model']['dropout']
        )

        self.proteinEmbedder = Embedder(
            inputDimension=hyperParameter['model']['enc']['hidden_dim'],
            hiddenDimension=hyperParameter['model']['enc']['hidden_dim'] * hyperParameter['model']['emb']['ffn_factor'],
            headNumber=hyperParameter['model']['emb']['head_num'],
            layerNumber=hyperParameter['model']['emb']['layer_num'],
            dropout=hyperParameter['model']['dropout']
        )
        if hyperParameter['model']['dec']['arch'] == 'sym':
            self.proteinClassifier = SymmetricTransformerDecoder(
                headNumber=hyperParameter['model']['dec']['head_num'],
                layerNumber=hyperParameter['model']['dec']['layer_num'],
                inputDimension=hyperParameter['model']['enc']['hidden_dim'],
                dropout=hyperParameter['model']['dropout']
            )
        elif hyperParameter['model']['dec']['arch'] == 'plain':
            self.proteinClassifier = PlainTransformerDecoder(
                headNumber=hyperParameter['model']['dec']['head_num'],
                layerNumber=hyperParameter['model']['dec']['layer_num'],
                inputDimension=hyperParameter['model']['enc']['hidden_dim'],
                dropout=hyperParameter['model']['dropout']
            )

        # loss
        self.bceLoss = nn.BCEWithLogitsLoss()
        self.crossEntropyLoss = nn.CrossEntropyLoss()
        if hyperParameter['model']['loss_type'] == 'InfoNCE':
            self.contrastiveLoss = model.Loss.PPIContrastiveLoss(hyperParameter['model']['loss_temperature'])
        elif hyperParameter['model']['loss_type'] == 'SigLIP':
            self.contrastiveLoss = model.Loss.PPISigLIPLoss()
        else:
            raise NotImplementedError
        self.interfaceLoss = model.Loss.InterfaceLoss()

        # use padding
        if "padding_interface" in hyperParameter:
            self.useZeroInterfacePadding = hyperParameter["padding_interface"]
        else:
            print(f"Missing arg for interface padding, default False")
            self.useZeroInterfacePadding = False
        print(f"Use zero interface padding: {self.useZeroInterfacePadding}")

        # val step
        self.val_step = 0

        # metric
        self.roc = mc.BinaryROC()
        self.auroc = mc.BinaryAUROC()
        self.pr = mc.BinaryPrecisionRecallCurve()

        self.cosineRoc = mc.BinaryROC()
        self.cosineAUROC = mc.BinaryAUROC()
        self.cosinePR = mc.BinaryPrecisionRecallCurve()

        self.predictions = []
        self.groundTruths = []

        self.cosineSimPrediction = []
        self.cosineSimGroundTruth = []

        # Inference Mode, embedding or classification
        self.inferenceMode = 'Embedding'
        self.dataFormat = 'Pairwise'

        # Train embedder or decoder only
        self.useDecoderOnly = False if "use_dec_only" not in hyperParameter else hyperParameter["use_dec_only"]
        self.useEmbedderOnly = False if "use_emb_only" not in hyperParameter else hyperParameter["use_emb_only"]

        print(f"Use Dec only:{self.useDecoderOnly}")
        print(f"Use Emb only:{self.useEmbedderOnly}")

        # loss factor control
        self.interfaceLossFactor = hyperParameter['interface_factor']
        self.contrastiveLossFactor = hyperParameter['contrastive_factor']
        self.interactionLossFactor = hyperParameter['interaction_factor']


        if self.useDecoderOnly:
            self.proteinEmbedder.requires_grad_(False)
            self.contrastiveLoss.requires_grad_(False)
        if self.useEmbedderOnly:
            self.proteinClassifier.requires_grad_(False)


        self.usablePairwiseMode = ['Embedding', 'All', 'Classification']
        self.usableMonomerMode = ['Encoding']

        self.usePadding = False
        self.paddingEmbedding: Embedding | None = None

        self.trainingMode = 'fullScale'
        self.embeddingOnce = False
        self._embeddingHistory = []

    def SetEmbeddingOnce(self, embeddingOnce: bool=True):
        self.embeddingOnce = embeddingOnce

    def SetTrainingMode(self, mode: Literal['fullScale', 'finetuneNbAgWithClsHeadOnly', 'finetuneEmbedder']):
        if mode not in ['fullScale', 'finetuneNbAgWithClsHeadOnly', 'finetuneEmbedder']:
            raise ValueError
        self.trainingMode = mode
        if mode == 'fullScale':
            return
        try:
            assert self.useDecoderOnly == False
            assert self.useEmbedderOnly == False
        except AssertionError:
            print('Finetuning mode can not operates with using decoder only or using embedder only')
            raise ValueError
        self.proteinEncoder.requires_grad_(False)
        if mode == 'finetuneNbAgWithClsHeadOnly':
            self.proteinEmbedder.requires_grad_(False)
            self.contrastiveLoss.requires_grad_(False)
            self.proteinEncoder.requires_grad_(False)
            self.proteinClassifier.FreezeCrossChainModule()
        if mode == 'finetuneEmbedder':
            self.proteinClassifier.requires_grad_(False)
            self.proteinEncoder.requires_grad_(False)
            if self.hyperParameter['model']['loss_type'] == 'InfoNCE':
                self.contrastiveLoss = model.Loss.PPIContrastiveLoss(self.hyperParameter['model']['loss_temperature'])
            elif self.hyperParameter['model']['loss_type'] == 'SigLIP':
                self.contrastiveLoss = model.Loss.PPISigLIPLoss()
            else:
                raise NotImplementedError

    def SetDataFormatToMonomerInput(self):
        self.dataFormat = 'Monomer'

    def SetDataFormatToPairwiseInput(self):
        self.dataFormat = 'Pairwise'

    def InferenceClassificationScore(self):
        self.inferenceMode = 'Classification'

    def InferenceEmbedding(self):
        self.inferenceMode = 'Embedding'

    def InferenceInterface(self):
        self.inferenceMode = 'Interface'

    def InferenceAll(self):
        self.inferenceMode = 'All'

    def InferenceEncoding(self):
        self.inferenceMode = 'Encoding'

    def EncodingProtein(self, inputProteinGraph):
        return self.proteinEncoder(inputProteinGraph)

    def EmbeddingProtein(self, inputEncodedSeq):
        return self.proteinEmbedder(inputEncodedSeq)

    def GetPredictionScore(self, inputEncodedSeqA, inputEncodedSeqB):
        return self.proteinClassifier.GetInteractionScore(inputEncodedSeqA, inputEncodedSeqB)

    def GetInferenceInterface(self, inputEncodedSeqA, inputEncodedSeqB):
        return self.proteinClassifier.GetInterface(inputEncodedSeqA, inputEncodedSeqB)

    def predict_step(self, batch, batch_idx):
        if self.dataFormat == 'Pairwise':
            feature, label = batch
            print(label)
            chosenMonomerA, chosenMonomerB = feature
            selectedMonomerA, selectedMonomerB = label[0][0], label[0][1]
            if self.inferenceMode == 'Embedding':
                embeddingA = self.EmbeddingProtein(chosenMonomerA)
                embeddingB = self.EmbeddingProtein(chosenMonomerB)
                return embeddingA, embeddingB
            elif self.inferenceMode == 'Classification':
                return self.GetPredictionScore(chosenMonomerA, chosenMonomerB)
            elif self.inferenceMode == 'Interface':
                return self.GetInferenceInterface(chosenMonomerA, chosenMonomerB)
            elif self.inferenceMode == 'All':
                if self.embeddingOnce:
                    if selectedMonomerA not in self._embeddingHistory:
                        embeddingA = self.EmbeddingProtein(chosenMonomerA)
                        self._embeddingHistory.append(selectedMonomerA)
                    else:
                        print(f'Already predicted embedding:{selectedMonomerA}')
                        embeddingA = None
                    if selectedMonomerB not in self._embeddingHistory:
                        embeddingB = self.EmbeddingProtein(chosenMonomerB)
                        self._embeddingHistory.append(selectedMonomerB)
                    else:
                        print(f'Already predicted embedding:{selectedMonomerB}')
                        embeddingB = None
                else:
                    embeddingA = self.EmbeddingProtein(chosenMonomerA)
                    embeddingB = self.EmbeddingProtein(chosenMonomerB)
                return embeddingA, embeddingB, self.GetPredictionScore(chosenMonomerA, chosenMonomerB)
        if self.dataFormat == 'Monomer':
            # feature, label
            monomerInputs, label = batch
            if self.inferenceMode == 'Encoding':
                outputEmbedding = self.proteinEncoder(monomerInputs)
                return outputEmbedding
            elif self.inferenceMode == 'Embedding':
                outputEmbedding = self.proteinEmbedder(monomerInputs)
                return outputEmbedding

        print(f"Invalid inference mode:{self.inferenceMode} for data format:{self.dataFormat} ")
        raise ValueError

    def GetGlobalL2(self):
        decayAppliedParameters = [
            par for name, par in self.named_parameters() if 'norm' not in name.lower() and par.requires_grad
        ]
        norms = []
        for par in decayAppliedParameters:
            l2 = par.data.norm(2).item() / sqrt(float(par.numel()))
            norms.append(l2)
        return sum(norms) / len(norms)


    def GlobalMonitoring(self, stepPerRecord: int=20):
        # Being used for training stage monitoring
        self.log("train_LR", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, on_epoch=True, on_step=True, logger=True)
        # Skipping those steps which are not needed to be recorded
        if self.global_step % stepPerRecord != 0:
            return
        # Get L2
        trainingL2 = self.GetGlobalL2()
        self.log('train_L2', trainingL2, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        if isinstance(self.contrastiveLoss, model.Loss.PPISigLIPLoss):
            self.log("loss_par_t", self.contrastiveLoss.learnableNormConstantTemperature, prog_bar=True, on_epoch=True, on_step=True, logger=True)
            self.log("loss_par_b", self.contrastiveLoss.learnableNormConstantB, prog_bar=True, on_epoch=True, on_step=True, logger=True)
            if self.contrastiveLoss.learnableNormConstantB.grad is not None:
                self.log('loss_par_b_grad', self.contrastiveLoss.learnableNormConstantB.grad.item(), prog_bar=True, on_epoch=True, on_step=True, logger=True)
            if self.contrastiveLoss.learnableNormConstantTemperature.grad is not None:
                self.log('loss_par_t_grad', self.contrastiveLoss.learnableNormConstantTemperature.grad.item(), prog_bar=True, on_epoch=True, on_step=True, logger=True)

    def training_step(self, batch, batch_idx):
        if self.trainingMode == 'fullScale':
            return self.FullScaleTraining(batch, batch_idx)
        elif self.trainingMode == 'finetuneEmbedder':
            return self.EmbedderFinetuneTraining(batch, batch_idx)
        elif self.trainingMode == 'finetuneNbAgWithClsHeadOnly':
            return self.NbAgDecoderFinetuneTraining(batch, batch_idx)
        else:
            raise NotImplementedError(f"{self.trainingMode} is not implemented for a training mode")


    def _ParsingBatch(self, batch):
        feature, label = batch
        positives, negatives, interfaces = feature
        positiveMonomers, negativeMonomers = label
        assert len(positives) % 2 == 0
        print(f"Selected positive monomers:{positiveMonomers}")
        print(f"Selected negative monomers:{negativeMonomers}")
        return positives, negatives, positiveMonomers, negativeMonomers, interfaces

    def NbAgDecoderFinetuneTraining(self, batch, batch_idx):
        positives, negatives, positiveMonomers, negativeMonomers, interfaces = self._ParsingBatch(batch)
        positiveBatchSize = len(positives) // 2
        scores = []
        labels = []
        for i in range(positiveBatchSize):
            # first nanobody, second antigen, be careful
            positiveNanobody = positiveMonomers[2 * i]
            positiveAntigen = positiveMonomers[2 * i + 1]
            positiveNanobodyEmbedding = positives[2 * i]
            positiveAntigenEmbedding = positives[2 * i + 1]
            # use antibody as query
            positivePairScores, interfaces = self.GetPredictionScore(positiveNanobodyEmbedding, positiveAntigenEmbedding)
            # Q score only
            score, _ = positivePairScores
            print(f"Selected antigen:{positiveAntigen} and positive nanobody:{positiveNanobody} with score:{score.item()}")
            scores.append(score)
            labels.append([1])

            # negatives
            for negativeIndex in range(len(negatives)):
                negativeNanobodyEmbedding = negatives[negativeIndex]
                negativeNanobody = negativeMonomers[negativeIndex]
                negativePairScores, interfaces = self.GetPredictionScore(negativeNanobodyEmbedding, positiveNanobodyEmbedding)
                score, _ = negativePairScores
                print(f"Selected antigen:{positiveAntigen} and negative nanobody:{negativeNanobody} with score:{score.item()}")
                scores.append(score)
                labels.append([0])
        scores = torch.cat(scores, dim=0)
        groundTruth = torch.tensor(labels, dtype=torch.float).to(device=scores.device)
        loss = self.bceLoss(scores, groundTruth)

        # log loss
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return {'loss': loss}


    def EmbedderFinetuneTraining(self, batch, batch_idx):
        positives, negatives, positiveMonomers, negativeMonomers, interfaces = self._ParsingBatch(batch)
        positiveBatchSize = len(positives) // 2
        positiveGroupA = []
        positiveGroupB = []
        negativeEmbeddings = []
        for i in range(positiveBatchSize):
            positiveMonomerA = positiveMonomers[2 * i]
            positiveMonomerB = positiveMonomers[2 * i + 1]
            positiveA = positives[2 * i]
            positiveB = positives[2 * i + 1]
            print(f"Selected positive pair:{positiveMonomerA}_{positiveMonomerB}")
            positiveEmbeddingA = self.EmbeddingProtein(positiveA)
            positiveEmbeddingB = self.EmbeddingProtein(positiveB)
            positiveGroupA.append(positiveEmbeddingA)
            positiveGroupB.append(positiveEmbeddingB)
            # negatives
        for negativeIndex in range(len(negatives)):
            negativeA = negatives[negativeIndex]
            negativeMonomer = negativeMonomers[negativeIndex]
            print(f"Using negative monomer:{negativeMonomer}")
            negativeEmbedding = self.EmbeddingProtein(negativeA)
            negativeEmbeddings.append(negativeEmbedding)
        contrastiveLoss, positiveSimilarity, negativeMeanSimilarity = self.contrastiveLoss(positiveGroupA,
                                                                                           positiveGroupB,
                                                                                           negativeEmbeddings)
        loss = contrastiveLoss * self.contrastiveLossFactor

        # log loss
        self.log('train_contra_loss', contrastiveLoss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('train_pos_sim', positiveSimilarity, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('train_neg_sim', negativeMeanSimilarity, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return {'loss': loss}


    def FullScaleTraining(self, batch, batch_idx):
        positives, negatives, positiveMonomers, negativeMonomers, interfacesGroundTruth = self._ParsingBatch(batch)
        positiveBatchSize = len(positives) // 2
        positiveGroupA = []
        positiveGroupB = []
        predictionScores = []
        groundTruthScores = []
        positiveProteinSequence = []
        predictionInterfaces = []
        for i in range(positiveBatchSize):
            positiveMonomerA = positiveMonomers[2 * i]
            positiveMonomerB = positiveMonomers[2 * i + 1]
            positiveA = positives[2 * i]
            positiveB = positives[2 * i + 1]
            print(f"Selected positive pair:{positiveMonomerA}_{positiveMonomerB}")
            # positive
            positiveProteinSeqA = self.EncodingProtein(positiveA)
            positiveProteinSeqB = self.EncodingProtein(positiveB)
            positiveProteinSequence.append(positiveProteinSeqA)
            positiveProteinSequence.append(positiveProteinSeqB)

            # embeddings
            if not self.useDecoderOnly:
                positiveEmbeddingA = self.EmbeddingProtein(positiveProteinSeqA)
                positiveEmbeddingB = self.EmbeddingProtein(positiveProteinSeqB)
                positiveGroupA.append(positiveEmbeddingA)
                positiveGroupB.append(positiveEmbeddingB)
                negativeEmbeddings = []
            else:
                print("Stop embedding cal: positive pairs")

            # classification
            if not self.useEmbedderOnly:
                positivePairScores, interfaces = self.proteinClassifier(positiveProteinSeqA, positiveProteinSeqB)
                proteinAScore, proteinBScore = positivePairScores
                proteinAInterface, proteinBInterface = interfaces
                predictionInterfaces.append(proteinAInterface)
                predictionInterfaces.append(proteinBInterface)
                if proteinAScore.item() < proteinBScore.item():
                    positivePairScore = proteinAScore
                else:
                    positivePairScore = proteinBScore
                print(f"Positive pair {positiveMonomerA}_{positiveMonomerB} logit {positivePairScore.item()} with {proteinAScore.item()}&{proteinBScore.item()}")
                predictionScores.append(positivePairScore)
                groundTruthScores.append([1])
            else:
                print("Stop cls cal: negative pairs")

        negativePredictionInterfaces = []
        negativeGroundTruthInterfaces = []
        # negatives
        for negativeIndex in range(len(negatives)):
            feature = negatives[negativeIndex]
            negativeMonomer = negativeMonomers[negativeIndex]

            negativeSeq = self.EncodingProtein(feature)
            # embedding
            if not self.useDecoderOnly:
                negativeEmbedding = self.EmbeddingProtein(negativeSeq)
                negativeEmbeddings.append(negativeEmbedding)
            else:
                print("Stop embedding cal: negative pairs")

            # classification
            if not self.useEmbedderOnly:
                for positiveIndex in range(len(positiveProteinSequence)):
                    positiveSequence = positiveProteinSequence[positiveIndex]
                    positiveMonomer = positiveMonomers[positiveIndex]
                    negativeScores, negativeInterfaces = self.proteinClassifier(positiveSequence, negativeSeq)
                    positiveSampleMonomerInterface, negativeSampleMonomerInterface = negativeInterfaces

                    # Physics inspired
                    zeroContactInterfaceA = torch.zeros([positiveSampleMonomerInterface.shape[-2], 5]).to(positiveSampleMonomerInterface)
                    zeroContactInterfaceB = torch.zeros([negativeSampleMonomerInterface.shape[-2], 5]).to(positiveSampleMonomerInterface)
                    zeroContactInterfaceA[:, 4] = 1
                    zeroContactInterfaceB[:, 4] = 1

                    negativePredictionInterfaces.append(positiveSampleMonomerInterface)
                    negativeGroundTruthInterfaces.append(zeroContactInterfaceA)

                    negativePredictionInterfaces.append(negativeSampleMonomerInterface)
                    negativeGroundTruthInterfaces.append(zeroContactInterfaceB)

                    # continue scores cal.

                    proteinAScore, proteinBScore = negativeScores

                    if proteinAScore.item() > proteinBScore.item():
                        negativeScore = proteinAScore
                    else:
                        negativeScore = proteinBScore

                    print(f"Negative pair {positiveMonomer}_{negativeMonomer} logit {negativeScore.item()} with {proteinAScore.item()}&{proteinBScore.item()}")
                    predictionScores.append(negativeScore)
                    groundTruthScores.append([0])
            else:
                print("Stop cls cal: negative pairs")

        # loss calculation
        if not self.useDecoderOnly:
            contrastiveLoss, positiveSimilarity, negativeMeanSimilarity = self.contrastiveLoss(positiveGroupA, positiveGroupB, negativeEmbeddings)
        else:
            contrastiveLoss, positiveSimilarity, negativeMeanSimilarity = (0.0, 1.0, -1.0)
            print("Stop embedding cal: loss")

        if not self.useEmbedderOnly:
            predictionScores = torch.cat(predictionScores, dim=0)
            groundTruth = torch.tensor(groundTruthScores, dtype=torch.float).to(device=predictionScores.device)
            classificationLoss = self.bceLoss(predictionScores, groundTruth)
            # interfaces loss
            lossFactor = len(predictionInterfaces)
            interfaceLoss = 0.0
            for interfaceIndex in range(lossFactor):
                predictedInterface = predictionInterfaces[interfaceIndex]
                groundTruthInterface = interfacesGroundTruth[interfaceIndex]
                interfaceLoss = interfaceLoss + self.interfaceLoss(predictedInterface, groundTruthInterface)

            if self.useZeroInterfacePadding:
                lossFactor = lossFactor + len(negativeGroundTruthInterfaces)
                for interfaceIndex in range(len(negativeGroundTruthInterfaces)):
                    predictedInterface = negativePredictionInterfaces[interfaceIndex]
                    groundTruthInterface = negativeGroundTruthInterfaces[interfaceIndex]
                    interfaceLoss = interfaceLoss + self.interfaceLoss(predictedInterface, groundTruthInterface)

            interfaceLoss = interfaceLoss / lossFactor
        else:
            classificationLoss = 0.0
            interfaceLoss = 0.0
            print("Stop cls cal: loss")

        factors = [self.interfaceLossFactor, self.contrastiveLossFactor, self.interactionLossFactor]
        losses = [interfaceLoss, contrastiveLoss, classificationLoss]
        loss = 0.0
        for i in range(3):
            if factors[i] == 0.0:
                continue
            else:
                loss = loss + factors[i] * losses[i]

        if (self.global_step % 300 == 0) and (not self.useEmbedderOnly):
            for i in range(positiveBatchSize):
                positiveMonomerA = positiveMonomers[2 * i]
                positiveMonomerB = positiveMonomers[2 * i + 1]
                predictedInterfaceA = predictionInterfaces[2 * i]
                predictedInterfaceB = predictionInterfaces[2 * i + 1]
                groundTruthInterfaceA = interfacesGroundTruth[2 * i]
                groundTruthInterfaceB = interfacesGroundTruth[2 * i + 1]

                title = f"{positiveMonomerA}_{positiveMonomerB}_A"
                self.LogInterface(predictedInterfaceA, groundTruthInterfaceA, title, f"TrainInterfacesA-{i}.png")
                title = f"{positiveMonomerA}_{positiveMonomerB}_B"
                self.LogInterface(predictedInterfaceB, groundTruthInterfaceB, title, f"TrainInterfacesB-{i}.png")

                if self.useZeroInterfacePadding:
                    negativeMonomer = negativeMonomers[0]
                    positiveMonomer = positiveMonomers[0]
                    positiveInterfaceSample = negativePredictionInterfaces[0]
                    negativeInterfaceSample = negativePredictionInterfaces[1]
                    positiveGroundTruthSample = negativeGroundTruthInterfaces[0]
                    negativeGroundTruthSample = negativeGroundTruthInterfaces[1]

                    title = f"{positiveMonomer}_{negativeMonomer}_A"
                    self.LogInterface(positiveInterfaceSample, positiveGroundTruthSample, title, f"TrainInterfaceNegativeA-{i}.png")
                    title = f"{positiveMonomer}_{negativeMonomer}_B"
                    self.LogInterface(negativeInterfaceSample, negativeGroundTruthSample, title, f"TrainInterfacePositiveB-{i}.png")

            plt.close("all")


        # log loss
        self.log('train_interface_loss', interfaceLoss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('train_bce_loss', classificationLoss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('train_contra_loss', contrastiveLoss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('train_pos_sim', positiveSimilarity, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('train_neg_sim', negativeMeanSimilarity, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return {'loss': loss}

    def LogInterface(self, predictedInterface, groundTruthInterface, title, fileName='PredictedInterface.png'):
        predictedInterface = predictedInterface.detach().cpu().clone().float()[0]
        groundTruthInterface = groundTruthInterface.detach().cpu().clone().float()
        # Get prediction result
        values, indices = torch.topk(predictedInterface, 1, -1)
        predictionResult = torch.zeros(predictedInterface.shape)
        predictionResult = predictionResult.scatter(1, indices, 1)
        predictionError = predictionResult - groundTruthInterface[:, 1:]

        fig, axes = plt.subplot_mosaic([['A'], ['B'], ['C'], ['D']], figsize=(predictedInterface.shape[0] * 0.08, 12))
        fig.suptitle(title)
        sns.heatmap(predictedInterface.transpose(0, 1), annot=False, cbar=True, yticklabels=False, ax=axes['A'])
        axes['A'].set_title('Predict Logits')
        sns.heatmap(groundTruthInterface.transpose(0, 1), annot=False, cbar=True, yticklabels=False, ax=axes['B'])
        axes['B'].set_title('Ground Truth')
        sns.heatmap(predictionResult.transpose(0, 1), annot=False, cbar=True, yticklabels=False, ax=axes['C'])
        axes['C'].set_title('Predicted Top-1 Result')
        sns.heatmap(predictionError.transpose(0, 1), annot=False, cbar=True, yticklabels=False, ax=axes['D'])
        axes['D'].set_title('Error Map')
        fig.tight_layout()
        self.logger.experiment.log_figure(fileName, plt.gcf(), step=self.global_step)
        plt.clf()
        plt.cla()


    def UpdateBinaryClassificationResult(self, predictionTensor, label: int):
        predictionTensor = predictionTensor.detach().to('cpu')
        if label == 0:
            labelTensor = torch.zeros([1, 1], dtype=torch.int, device='cpu')
        elif label == 1:
            labelTensor = torch.ones([1, 1], dtype=torch.int, device='cpu')
        else:
            raise ValueError
        self.roc.update(predictionTensor, labelTensor)
        self.auroc.update(predictionTensor, labelTensor)
        self.pr.update(predictionTensor, labelTensor)
        predictionTensor = F.sigmoid(predictionTensor)
        self.predictions.append(predictionTensor)
        self.groundTruths.append(label)

    def UpdateCosineSimilarityResult(self, embeddingA, embeddingB, label: int):
        # print(embeddingA.shape)
        predictionTensor = torch.nn.functional.cosine_similarity(embeddingA, embeddingB, dim=-1).unsqueeze(0)
        predictionTensor = predictionTensor.detach().to('cpu')
        # print(predictionTensor)
        if label == 0:
            labelTensor = torch.zeros([1, 1], dtype=torch.int, device='cpu')
        elif label == 1:
            labelTensor = torch.ones([1, 1], dtype=torch.int, device='cpu')
        else:
            raise ValueError
        self.cosineRoc.update(predictionTensor, labelTensor)
        self.cosineAUROC.update(predictionTensor, labelTensor)
        self.cosinePR.update(predictionTensor, labelTensor)
        # Let us just consider those sim < 0.0 is just negatives and set it to 0.0
        # so that the conf mat will be shown correctly
        if predictionTensor.item() < 0.0:
            predictionTensor = torch.Tensor([[0.0]]).bfloat16()
        self.cosineSimPrediction.append(predictionTensor)
        self.cosineSimGroundTruth.append(label)

    def validation_step(self, batch, batch_idx):
        feature, label = batch
        positives, negatives, groundTruthInterface = feature
        positivesMonomer, negativeMonomers = label

        positiveMonomerA, positiveMonomerB = positivesMonomer
        positiveA, positiveB = positives

        print(f"Selected positive pair:{positiveMonomerA}_{positiveMonomerB}")
        print(f"Selected negative monomers:{negativeMonomers}")

        # positive
        # print("Encoding Protein...")
        if self.trainingMode == 'fullScale':
            positiveProteinSeqA = self.EncodingProtein(positiveA)
            positiveProteinSeqB = self.EncodingProtein(positiveB)
        elif self.trainingMode == 'finetuneEmbedder' or self.trainingMode == 'finetuneNbAgWithClsHeadOnly':
            positiveProteinSeqA = positiveA
            positiveProteinSeqB = positiveB
        else:
            raise ValueError

        # print(f"Encoding shape:{positiveProteinSeqB.shape}")

        # print("Running classification of positive pairs...")
        positivePairScores, interface = self.GetPredictionScore(positiveProteinSeqA, positiveProteinSeqB)
        interfaceA, interfaceB = interface
        interfaceGRA, interfaceGRB = groundTruthInterface
        proteinAScore, proteinBScore = positivePairScores
        if self.trainingMode == 'fullScale' or self.trainingMode == 'finetuneEmbedder':
            positivePairScore = (proteinAScore + proteinBScore) / 2.0
            print(
                f"Positive pair {positiveMonomerA}_{positiveMonomerB} logit "
                f"{positivePairScore.item()} with {proteinAScore.item()}&{proteinBScore.item()}"
            )
        elif self.trainingMode == 'finetuneNbAgWithClsHeadOnly':
            positivePairScore = proteinAScore
            print(
                f"Positive pair {positiveMonomerA}_{positiveMonomerB} logit "
                f"{positivePairScore.item()}"
            )
        else:
            raise ValueError
        # print(f"Output shape:{positivePairScore.shape}")


        # update positive pair
        self.UpdateBinaryClassificationResult(positivePairScore, 1)

        positiveEmbeddingA = self.EmbeddingProtein(positiveProteinSeqA)
        positiveEmbeddingB = self.EmbeddingProtein(positiveProteinSeqB)
        self.UpdateCosineSimilarityResult(positiveEmbeddingA, positiveEmbeddingB, 1)

        # negatives
        if self.trainingMode == 'fullScale' or self.trainingMode == 'finetuneEmbedder':
            negativeEmbeddings = []
            predictionScores = [positivePairScore]
            groundTruthScores = [[1]]
            for index in range(len(negatives)):
                feature = negatives[index]
                negativeMonomer = negativeMonomers[index]
                if self.trainingMode == 'fullScale':
                    negativeSeq = self.EncodingProtein(feature)
                elif self.trainingMode == 'finetuneEmbedder':
                    negativeSeq = feature
                else:
                    raise ValueError
                # embedding
                negativeEmbedding = self.EmbeddingProtein(negativeSeq)
                self.UpdateCosineSimilarityResult(negativeEmbedding, positiveEmbeddingA, 0)
                self.UpdateCosineSimilarityResult(negativeEmbedding, positiveEmbeddingB, 0)

                negativeEmbeddings.append(negativeEmbedding)
                # classification
                negativeScoreA, negativeInterfaceA = self.proteinClassifier(positiveProteinSeqA, negativeSeq)
                negativeScoreB, negativeInterfaceB = self.proteinClassifier(positiveProteinSeqB, negativeSeq)

                negativeBinderASample, _ = negativeInterfaceA
                negativeBinderBSample, _ = negativeInterfaceB

                proteinAScoreOfA, proteinBScoreOfA = negativeScoreA
                proteinAScoreOfB, proteinBScoreOfB = negativeScoreB

                negativeScoreA = (proteinAScoreOfA + proteinBScoreOfA) / 2.0
                negativeScoreB = (proteinAScoreOfB + proteinBScoreOfB) / 2.0

                print(
                    f"Negative pair {positiveMonomerA}_{negativeMonomer} logit {negativeScoreA.item()} with {proteinAScoreOfA.item()}&{proteinBScoreOfA.item()}")
                print(
                    f"Negative pair {positiveMonomerB}_{negativeMonomer} logit {negativeScoreB.item()} with {proteinAScoreOfB.item()}&{proteinBScoreOfB.item()}")

                predictionScores.append(negativeScoreA)
                predictionScores.append(negativeScoreB)

                self.UpdateBinaryClassificationResult(negativeScoreA, 0)
                self.UpdateBinaryClassificationResult(negativeScoreB, 0)

                groundTruthScores.append([0])
                groundTruthScores.append([0])

            # loss calculation
            contrastiveLoss, positiveSimilarity, negativeMeanSimilarity = self.contrastiveLoss([positiveEmbeddingA],
                                                                                               [positiveEmbeddingB],
                                                                                               negativeEmbeddings)
        elif self.trainingMode == 'finetuneNbAgWithClsHeadOnly':
            negativeEmbeddings = []
            predictionScores = [positivePairScore]
            groundTruthScores = [[1]]
            for index in range(len(negatives)):
                feature = negatives[index]
                negativeMonomer = negativeMonomers[index]
                negativeSeq = feature
                # embedding
                negativeEmbedding = self.EmbeddingProtein(negativeSeq)
                self.UpdateCosineSimilarityResult(negativeEmbedding, positiveEmbeddingB, 0)
                negativeEmbeddings.append(negativeEmbedding)
                # classification
                negativeScoreA, negativeInterfaceA = self.proteinClassifier(negativeSeq, positiveProteinSeqB)

                negativeBinderASample, _ = negativeInterfaceA

                proteinAScoreOfA, proteinBScoreOfA = negativeScoreA

                negativeScoreA = proteinAScoreOfA

                print(
                    f"Negative pair NB_AG: {negativeMonomer}_{positiveMonomerB} logit {negativeScoreA.item()}")


                predictionScores.append(negativeScoreA)

                self.UpdateBinaryClassificationResult(negativeScoreA, 0)

                groundTruthScores.append([0])

            # loss calculation
            contrastiveLoss, positiveSimilarity, negativeMeanSimilarity = self.contrastiveLoss([positiveEmbeddingB],
                                                                                               [positiveEmbeddingB],
                                                                                               negativeEmbeddings)
        else:
            raise NotImplementedError


        if self.trainingMode == 'fullScale':
            interfaceLoss = (self.interfaceLoss(interfaceA, interfaceGRA) + self.interfaceLoss(interfaceB, interfaceGRB)) / 2.0
            # Log interface
            if self.val_step % 60 == 0:
                self.LogInterface(interfaceA, interfaceGRA, f'{positiveMonomerA}-{positiveMonomerB}-A',
                                  'Val-Interface-A.png')

                self.LogInterface(negativeBinderASample, interfaceGRA, f'{positiveMonomerA}-FalseMap', 'Val-Interface-False-A.png')
                self.LogInterface(negativeBinderBSample, interfaceGRB, f'{positiveMonomerB}-FalseMap', 'Val-Interface-False-B.png')

                self.LogInterface(interfaceB, interfaceGRB, f'{positiveMonomerA}-{positiveMonomerB}-B',
                                  'Val-Interface-B.png')
        else:
            interfaceLoss = 0.0
        predictionScores = torch.cat(predictionScores, dim=0)
        groundTruth = torch.tensor(groundTruthScores, dtype=torch.float).to(device=predictionScores.device)
        classificationLoss = self.bceLoss(predictionScores, groundTruth)
        loss = contrastiveLoss + classificationLoss + interfaceLoss

        # log loss
        self.log('val_interface_loss', interfaceLoss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('val_bce_loss', classificationLoss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('val_contra_loss', contrastiveLoss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('val_pos_sim', positiveSimilarity, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('val_neg_sim', negativeMeanSimilarity, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.val_step = self.val_step + 1

    def LogConfusionMatrixAndResetLists(self, predictionsTensorList, groundTruthTensorList, thresholds, fileName):
        predictions = torch.cat(predictionsTensorList, dim=-1).to('cpu')
        thresholdsForConfusionMat = thresholds
        fig, axs = plt.subplots(1, len(thresholdsForConfusionMat), figsize=(5 * len(thresholdsForConfusionMat), 5))
        predictions = predictions[0]
        targets = torch.tensor(groundTruthTensorList).to('cpu')
        for index in range(len(thresholdsForConfusionMat)):
            th = thresholdsForConfusionMat[index]
            ax = axs[index]
            confMat = mc.BinaryConfusionMatrix(threshold=th)
            confMat.update(predictions, targets)
            confMat.plot(ax=ax)
            ax.set_title(f'{th}')
        plt.tight_layout()
        self.logger.experiment.log_figure(fileName, plt.gcf(), step=self.global_step)
        plt.clf()
        plt.cla()
        predictionsTensorList.clear()
        groundTruthTensorList.clear()

    def LogROCCurve(self, rocObject, curveName):
        # ROC curve
        fpr, tpr, thres = rocObject.compute()
        plt.plot(fpr.cpu(), tpr.cpu())
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        self.logger.experiment.log_figure(f"{curveName}.png", plt.gcf(), step=self.global_step)
        plt.clf()
        plt.cla()
        self.logger.experiment.log_curve(curveName, fpr.tolist(), tpr.tolist(), step=self.global_step)
        rocObject.reset()

    def LogAUROC(self, aurocObject, varName):
        self.log(varName, aurocObject.compute(), on_epoch=True, on_step=False, logger=True)
        aurocObject.reset()

    def LogPRCurve(self, prObject, fileName):
        # PR curve
        p, r, thres = prObject.compute()
        plt.plot(p.cpu(), r.cpu())
        plt.plot([0, 1], linewidth=0.5, alpha=1, linestyle='--')
        plt.title("PR Curve")
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()

        self.logger.experiment.log_figure(f"{fileName}.png", plt.gcf(), step=self.global_step)
        plt.clf()
        plt.cla()
        self.logger.experiment.log_curve(fileName, p.tolist(), r.tolist(), step=self.global_step)
        prObject.reset()
        # AUPR
        self.log(f'{fileName}-AUC', torch.abs(torch.trapezoid(y=p, x=r)), on_epoch=True, on_step=False)

    def LogGlobalDist(self, supervisedDistribution, contrastiveDistribution, supervisedLabels, contrastiveLabels, fileName):
        fig, axes = plt.subplot_mosaic([['A', 'A', 'B', 'C'], ['A', 'A', 'D', 'E']], figsize=(20, 10))
        # Handle datas
        assert supervisedLabels == contrastiveLabels
        # supP, supN, contP, contN
        scoresList = [[], [], [], []]
        for listIndex in range(len(supervisedDistribution)):
            if supervisedLabels[listIndex] == 1:
                scoresList[0].append(supervisedDistribution[listIndex])
            else:
                scoresList[1].append(supervisedDistribution[listIndex])
            if contrastiveLabels[listIndex] == 1:
                scoresList[2].append(contrastiveDistribution[listIndex])
            else:
                scoresList[3].append(contrastiveDistribution[listIndex])
        supervisedDistribution = torch.cat(supervisedDistribution, dim=-1).to('cpu')[0].float()
        contrastiveDistribution = torch.cat(contrastiveDistribution, dim=-1).to('cpu')[0].float()
        sns.histplot(x=supervisedDistribution, y=contrastiveDistribution, bins=20, cbar=True, ax=axes['A'])
        axes['A'].set_title('Joint Dist')
        axes['A'].set_xlabel('sup')
        axes['A'].set_ylabel('cont')
        sns.histplot(x=torch.cat(scoresList[0], dim=-1).to('cpu')[0].float(), bins=20, ax=axes['B'])
        axes['B'].set_title('Sup. Branch Positive Dist')
        axes['B'].set_xlabel('Scores')
        sns.histplot(x=torch.cat(scoresList[1], dim=-1).to('cpu')[0].float(), bins=20, ax=axes['C'])
        axes['C'].set_title('Sup. Branch Negative Dist')
        axes['C'].set_xlabel('Scores')
        sns.histplot(x=torch.cat(scoresList[2], dim=-1).to('cpu')[0].float(), bins=20, ax=axes['D'])
        axes['D'].set_title('Cont. Branch Positive Dist')
        axes['D'].set_xlabel('Scores')
        sns.histplot(x=torch.cat(scoresList[3], dim=-1).to('cpu')[0].float(), bins=20, ax=axes['E'])
        axes['E'].set_title('Cont. Branch Negative Dist')
        axes['E'].set_xlabel('Scores')
        fig.tight_layout()
        self.logger.experiment.log_figure(f"{fileName}.png", fig, step=self.global_step)

    def on_validation_epoch_end(self) -> None:
        # topk rate
        self.val_step = 0
        predictions = torch.cat(self.predictions, dim=-1).to('cpu')
        topkArrange = self.hyperParameter["log_topk_arrange"]
        for k in topkArrange:
            scores, indexes = torch.topk(predictions[0], k)
            truePositive = 0
            for index in indexes.tolist():
                if self.groundTruths[index] == 1.0:
                    truePositive += 1
            print(f"Top{k}:tp {truePositive}")
            precision = float(truePositive) / float(k)
            self.log(f'interaction_top_{k}_precision', precision, on_epoch=True, on_step=False, logger=True)

        # topk recall
        predictionsCosine = torch.cat(self.cosineSimPrediction, dim=-1).to('cpu')
        topkCosineArrange = self.hyperParameter["log_topk_cosine"]
        for k in topkCosineArrange:
            scores, indexes = torch.topk(predictionsCosine[0], k)
            truePositive = 0
            for index in indexes.tolist():
                if self.cosineSimGroundTruth[index] == 1.0:
                    truePositive += 1
            print(f"Top{k} cosine:tp {truePositive}")
            recall = float(truePositive)
            self.log(f'cosine_sim_top_{k}_recall', recall, on_epoch=True, on_step=False, logger=True)

        # Log dist
        self.LogGlobalDist(self.predictions, self.cosineSimPrediction, self.groundTruths, self.cosineSimGroundTruth, 'dist')

        # conf mat for binary score
        self.LogConfusionMatrixAndResetLists(
            self.predictions, self.groundTruths,
            self.hyperParameter["log_confusion_matrix_threshold"], "confMatBinary.png"
        )

        self.LogConfusionMatrixAndResetLists(
            self.cosineSimPrediction, self.cosineSimGroundTruth,
            self.hyperParameter["log_confusion_matrix_cosine_threshold"], "confMatCosineSim.png"
        )

        # ROC curve
        self.LogROCCurve(self.roc, 'BinaryDecoderROC')
        self.LogROCCurve(self.cosineRoc, 'CosineSimROC')

        # AUROC
        self.LogAUROC(self.cosineAUROC, 'CosineSimAUROC')
        self.LogAUROC(self.auroc, 'BinaryDecoderAUROC')

        # PR curve
        self.LogPRCurve(self.pr, 'BinaryDecoderPRCurve')
        self.LogPRCurve(self.cosinePR, 'CosineSimPRCurve')

        # L2
        l2 = self.GetGlobalL2()
        self.log(f'val_L2', l2, on_epoch=True, on_step=False, logger=True)

    def configure_optimizers(self):
        print('Configuring Opt.')
        if self.hyperParameter["optimizer_config"]["optimizer"] == "AdamW":
            print(f"Use AdamW optimizer")
            optimizer = torch.optim.AdamW(
                [
                    {
                        'params': [
                            par for name, par in self.named_parameters() if 'norm' in name.lower() and par.requires_grad
                                                                            and 'contrastiveLoss' not in name
                        ],
                        'weight_decay': 0.0,
                        'lr': self.hyperParameter["optimizer_config"]["learning_rate"],
                    },
                    {
                        'params': [
                            par for name, par in self.named_parameters() if
                            'norm' not in name.lower() and par.requires_grad
                        ],
                        'weight_decay': self.hyperParameter["optimizer_config"]["weight_decay"],
                        'lr': self.hyperParameter["optimizer_config"]["learning_rate"],
                    },
                    {
                        'params': [
                            par for name, par in self.named_parameters() if 'norm' in name.lower() and par.requires_grad
                                                                            and 'contrastiveLoss' in name
                        ],
                        'lr': self.hyperParameter["optimizer_config"]["learning_rate"] * 1000,
                        'weight_decay': 0.0,
                    },
                ],
                amsgrad=True
            )
        elif self.hyperParameter["optimizer_config"]["optimizer"] == "Muon":
            blockOutputName = 'proteinClassifier.outputLayer'
            optimizer = Muon(
                params=[
                    {
                        'params': [
                            par for name, par in self.named_parameters() if
                            'norm' not in name.lower() and par.requires_grad and blockOutputName not in name
                        ],
                        'weight_decay': self.hyperParameter["optimizer_config"]["weight_decay"],
                        'use_muon': True,
                    },
                    {
                        'params': [
                            par for name, par in self.named_parameters() if 'norm' in name.lower() and par.requires_grad
                        ],
                        'weight_decay': 0.0,
                        'use_muon': False,
                    },
                    {
                        'params': [
                            par for name, par in self.named_parameters() if
                            'norm' not in name.lower() and par.requires_grad and blockOutputName in name
                        ],
                        'adamw_wd': self.hyperParameter["optimizer_config"]["weight_decay_adam"],
                        'use_muon': False,
                    }
                ],
                lr=self.hyperParameter["optimizer_config"]["learning_rate"],
                adamw_lr=self.hyperParameter["optimizer_config"]["learning_rate_adam"],
            )
        else:
            print(f"Invalid optimizer: {self.hyperParameter['optimizer_config']['optimizer']}")
            raise NotImplementedError

        warmUpSteps = self.hyperParameter["optimizer_config"]["warmup_steps"]
        totalSteps = self.hyperParameter["optimizer_config"]["total_steps"]
        warmUpScheduler = LinearLR(
            optimizer=optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmUpSteps,
        )
        cosineAnnealingSchduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=totalSteps+warmUpSteps,
            eta_min=self.hyperParameter["optimizer_config"]["min_lr"]
        )
        sequentialScheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmUpScheduler, cosineAnnealingSchduler],
            milestones=[warmUpSteps]
        )
        scheduler = {
            "scheduler": sequentialScheduler,
            "interval": "step",
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure=None) -> None:
        self.GlobalMonitoring()
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)