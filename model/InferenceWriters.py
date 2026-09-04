from typing import Literal, Sequence, Any

import pandas as pd
import torch.nn
from lightning.pytorch.callbacks import BasePredictionWriter
import orjson
from lightning.pytorch.loggers import CometLogger
from sklearn.metrics import precision_recall_curve, roc_curve, auc

class UniprotProteomeEmbeddingWriter(BasePredictionWriter):

    def __init__(self, outputEmbeddingFile, write_interval, inputType='Monomer'):
        super().__init__(write_interval)
        self.outputs = {}
        self.filePath = outputEmbeddingFile
        self.inputType = inputType

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):

        if self.inputType == 'Monomer':
            feature, label = batch
            selectedMonomer = label[0]
            embedding = prediction
            embedding = embedding.float().cpu().numpy().tolist()
            self.outputs.update({selectedMonomer: embedding})

    def write_on_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            predictions,
            batch_indices,
    ) -> None:
        # print(self.resultDict)
        with open(self.filePath, 'wb') as file:
            file.write(orjson.dumps(self.outputs))


class UniprotProteomeEncodingWriter(BasePredictionWriter):

    def __init__(self, outputDir, write_interval, inputType: Literal['Pairwise', 'Monomer']='Monomer'):
        super().__init__(write_interval)
        # self.outputs = {}
        self.outputDir = outputDir
        self.filePath = outputDir
        self.inputType = inputType

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        # predictions
        if self.inputType == 'Monomer':
            feature, label = batch
            selectedMonomer = label[0]
            embedding = prediction
            # print(selectedMonomer)
            torch.save(embedding.to('cpu'), f"{self.outputDir}/{selectedMonomer}")

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:
        return None

class EncodingWriter(BasePredictionWriter):

    def __init__(self, outputDir, write_interval, inputType: Literal['Pairwise', 'Monomer']='Monomer'):
        super().__init__(write_interval)
        # self.outputs = {}
        self.outputDir = outputDir
        self.filePath = outputDir
        self.inputType = inputType

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        # predictions
        if self.inputType == 'Monomer':
            feature, label = batch
            selectedMonomer = label
            embedding = prediction
            selectedMonomer = selectedMonomer[0].split('-')
            selectedMonomer = f"{selectedMonomer[0]}-{selectedMonomer[1]}-{selectedMonomer[3]}"
            # print(selectedMonomer)
            torch.save(embedding.to('cpu'), f"{self.outputDir}/{selectedMonomer}")

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:
        return None

class EmbeddingWriter(BasePredictionWriter):

    def __init__(self, outputEmbeddingFile, write_interval, inputType: Literal['Pairwise', 'Monomer']='Pairwise'):
        super().__init__(write_interval)
        self.outputs = {}
        self.filePath = outputEmbeddingFile
        self.inputType = inputType

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):

        # predictions
        if self.inputType == 'Pairwise':
            feature, label = batch
            selectedMonomerA, selectedMonomerB, _ = label
            # entry, assembly, entityID, chainID, clusterID = id.split('-')
            selectedMonomerA = selectedMonomerA[0].split('-')
            selectedMonomerA = f"{selectedMonomerA[0]}-{selectedMonomerA[1]}-{selectedMonomerA[3]}"
            selectedMonomerB = selectedMonomerB[0].split('-')
            selectedMonomerB = f"{selectedMonomerB[0]}-{selectedMonomerB[1]}-{selectedMonomerB[3]}"
            embeddingA, embeddingB = prediction

            # to output format
            embeddingA = embeddingA.float().cpu().numpy().tolist()
            embeddingB = embeddingB.float().cpu().numpy().tolist()
            self.outputs.update({selectedMonomerA: embeddingA, selectedMonomerB: embeddingB})

        if self.inputType == 'Monomer':
            feature, label = batch
            selectedMonomer = label
            embedding = prediction
            selectedMonomer = selectedMonomer[0].split('-')
            selectedMonomer = f"{selectedMonomer[0]}-{selectedMonomer[1]}-{selectedMonomer[3]}"
            embedding = embedding.float().cpu().numpy().tolist()
            self.outputs.update({selectedMonomer: embedding})

    def write_on_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            predictions,
            batch_indices,
    ) -> None:
        # print(self.resultDict)
        with open(self.filePath, 'wb') as file:
            file.write(orjson.dumps(self.outputs))


class ClassificationWriter(BasePredictionWriter):

    def __init__(self, outputScoreFile, write_interval):
        super().__init__(write_interval)
        self.outputs = {}
        self.filePath = outputScoreFile

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):

        # predictions
        feature, label = batch
        selectedMonomerA, selectedMonomerB = label[0]
        scoreA = prediction[0][0].float().cpu().numpy().tolist()
        scoreB = prediction[0][1].float().cpu().numpy().tolist()

        # to output format
        self.outputs.update({f"{selectedMonomerA}:{selectedMonomerB}": (scoreA, scoreB)})

    def write_on_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            predictions,
            batch_indices,
    ) -> None:
        # print(self.resultDict)
        with open(self.filePath, 'wb') as file:
            file.write(orjson.dumps(self.outputs))


class BatchInferenceWriter(BasePredictionWriter):

    def _isPositivePair(self, pair: str):
        monomerA, monomerB = pair.split(':')
        if f"{monomerA}:{monomerB}" in self.positivePairList:
            return True
        elif f"{monomerB}:{monomerA}" in self.positivePairList:
            return True
        elif monomerA.split('-')[0] == monomerB.split('-')[0]:
            return True
        else:
            return False

    def __init__(self, cometLogger: CometLogger, write_interval, modelName, batchStep, datasetCSV: str, globalKValues,
                 crossKValues, nearestRating):
        super().__init__(write_interval)
        self.scoresOutputs = {}
        self.embeddingOutputs = {}
        self.modelName = modelName
        self.batchStep = batchStep
        self.ppiDataframe = pd.read_csv(datasetCSV, na_filter=False)
        length = len(self.ppiDataframe['entry'])
        self.ppi_df = self.ppiDataframe
        self.positivePairList = []
        for index in range(length):
            entry = str(self.ppi_df['entry'][index])
            assembly = str(self.ppi_df['assembly'][index])
            entityIDA = str(self.ppi_df['entityIDA'][index])
            entityIDB = str(self.ppi_df['entityIDB'][index])
            authorIDA = str(self.ppi_df['authorIDA'][index])
            authorIDB = str(self.ppi_df['authorIDB'][index])
            clusterIDA = int(self.ppi_df['clusterIDA'][index])
            clusterIDB = int(self.ppi_df['clusterIDB'][index])
            monomerA = f"{entry}-{assembly}-{entityIDA}-{authorIDA}-{clusterIDA}"
            monomerB = f"{entry}-{assembly}-{entityIDB}-{authorIDB}-{clusterIDB}"
            self.positivePairList.append(f"{monomerA}:{monomerB}")
        self.positivePairsEmbeddingSimilarities = []
        self.negativePairsEmbeddingSimilarities = []
        self.globalKValues = globalKValues

        self.remoteLogger = cometLogger
        # (emb, bin)
        self.crossTopRating = crossKValues
        self.nearestKStrategyRating = nearestRating


    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):

        # predictions
        feature, label = batch
        selectedMonomerA, selectedMonomerB = label
        embeddingA, embeddingB, score = prediction

        score = score.float().cpu().numpy().tolist()
        pairLabel = f"{selectedMonomerA[0]}:{selectedMonomerB[0]}"
        print(f'Current pair:{pairLabel}')

        # output logits
        self.scoresOutputs.update({pairLabel: score})
        # output embeddings
        if selectedMonomerA[0] not in self.embeddingOutputs:
            embeddingA = embeddingA.float().cpu().numpy().tolist()
            self.embeddingOutputs.update({selectedMonomerA[0]: embeddingA})
        if selectedMonomerB[0] not in self.embeddingOutputs:
            embeddingB = embeddingB.float().cpu().numpy().tolist()
            self.embeddingOutputs.update({selectedMonomerB[0]: embeddingB})

    def write_on_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            predictions,
            batch_indices,
    ) -> None:

        # Data preparation
        with open("comet_embeddings_tmp", 'wb') as file:
            file.write(orjson.dumps(self.embeddingOutputs))

        with open("comet_score_tmp", "wb") as file:
            file.write(orjson.dumps(self.scoresOutputs))

        pairScores = self.scoresOutputs
        scores = []
        labels = []
        totalPositive = 0
        for pair in pairScores:
            interaction = self._isPositivePair(pair)
            score = pairScores[pair][0][0]
            scores.append(score)
            labels.append(1.0 if interaction else 0.0)
            if interaction:
                totalPositive += 1
        print(f"TOTAL POSITIVE NUMBER: {totalPositive}")

        # Upload files
        self.remoteLogger.experiment.log_asset(
            file_data="comet_embeddings_tmp",
            file_name=f"{self.modelName}-{self.batchStep}-embeddings.json",
        )

        self.remoteLogger.experiment.log_asset(
            file_data="comet_score_tmp",
            file_name=f"{self.modelName}-{self.batchStep}-scores.json",
        )

        # Classification metrics
        # Calculate ROC & PRC Curve
        precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
        fpr, tpr, th = roc_curve(labels, scores, pos_label=1)
        self.remoteLogger.experiment.log_curve(
            name="cls-ROC-Curve",
            x=fpr,
            y=tpr,
            overwrite=False,
            step=self.batchStep,
        )
        self.remoteLogger.experiment.log_curve(
            name="cls-PR-Curve",
            x=recall,
            y=precision,
            overwrite=False,
            step=self.batchStep,
        )

        # Calculate AUROC & AUPR
        auprc = auc(recall, precision)
        auroc = auc(fpr, tpr)
        self.remoteLogger.experiment.log_metric(
            name='cls-AUPRC',
            value=auprc,
            step=self.batchStep,
            epoch=self.batchStep,
        )
        self.remoteLogger.experiment.log_metric(
            name='cls-AUROC',
            value=auroc,
            step=self.batchStep,
            epoch=self.batchStep,
        )

        # Calculate TOP-Ks for Binary score
        for kValue in self.globalKValues:

            topScores, indexes = torch.topk(torch.tensor(scores), kValue)
            hit = 0
            for index in indexes:
                if labels[index] == 1:
                    hit += 1
            topKPrecision = hit / kValue

            self.remoteLogger.experiment.log_metric(
                f"top-{kValue}-precision-binary",
                topKPrecision,
                step=self.batchStep,
                epoch=self.batchStep,
            )


        # embedding metrics
        embeddingScoresOutput = {}
        for pairLabel in self.scoresOutputs:
            monomerA, monomerB = pairLabel.split(':')
            embeddingA = torch.Tensor(self.embeddingOutputs[monomerA])
            embeddingB = torch.Tensor(self.embeddingOutputs[monomerB])
            interaction = self._isPositivePair(pairLabel)
            cosineSimilarity = torch.nn.functional.cosine_similarity(embeddingA, embeddingB,dim=-1).cpu().float().item()
            embeddingScoresOutput.update({pairLabel: cosineSimilarity})
            if interaction:
                self.positivePairsEmbeddingSimilarities.append(cosineSimilarity)
            else:
                self.negativePairsEmbeddingSimilarities.append(cosineSimilarity)


        # Calculate mean similarity
        positiveMean = torch.mean(torch.tensor(self.positivePairsEmbeddingSimilarities)).item()
        negativeMean = torch.mean(torch.tensor(self.negativePairsEmbeddingSimilarities)).item()
        self.remoteLogger.experiment.log_metric(
            f"positive-sim-mean",
            positiveMean,
            step=self.batchStep,
            epoch=self.batchStep,
        )
        self.remoteLogger.experiment.log_metric(
            f"negative-sim-mean",
            negativeMean,
            step=self.batchStep,
            epoch=self.batchStep,
        )

        scores = []
        labels = []
        for positiveScore in self.positivePairsEmbeddingSimilarities:
            scores.append(positiveScore)
            labels.append(1)

        for negativeScore in self.negativePairsEmbeddingSimilarities:
            scores.append(negativeScore)
            labels.append(0)

        # Calculate ROC & PRC Curve
        precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
        fpr, tpr, th = roc_curve(labels, scores, pos_label=1)
        self.remoteLogger.experiment.log_curve(
            name="emb-ROC-Curve",
            x=fpr,
            y=tpr,
            overwrite=False,
            step=self.batchStep,
        )
        self.remoteLogger.experiment.log_curve(
            name="emb-PR-Curve",
            x=recall,
            y=precision,
            overwrite=False,
            step=self.batchStep,
        )

        # Calculate AUROC & AUPR
        auprc = auc(recall, precision)
        auroc = auc(fpr, tpr)
        self.remoteLogger.experiment.log_metric(
            name='emb-AUPRC',
            value=auprc,
            step=self.batchStep,
            epoch=self.batchStep,
        )
        self.remoteLogger.experiment.log_metric(
            name='emb-AUROC',
            value=auroc,
            step=self.batchStep,
            epoch=self.batchStep,
        )

        # TOP-L/K
        embeddingLabels = []
        embeddingScores = []

        for label in embeddingScoresOutput:
            embeddingScores.append(embeddingScoresOutput[label])
            embeddingLabels.append(label)

        for topkl in self.crossTopRating:
            topInEmbedding, topInBinary = topkl
            topScores, indexes = torch.topk(torch.tensor(embeddingScores), topInEmbedding)
            hit = 0
            hitLabels = []
            for index in indexes:
                if self._isPositivePair(embeddingLabels[index]):
                    hit += 1
                hitLabels.append(embeddingLabels[index])
            topKPrecision = hit / topInEmbedding

            self.remoteLogger.experiment.log_metric(
                f"top-{topInEmbedding}-precision-emb",
                topKPrecision,
                step=self.batchStep,
                epoch=self.batchStep,
            )

            binaryScores = []
            binaryLabels = []

            for label in hitLabels:
                binaryScores.append(pairScores[label][0][0])
                binaryLabels.append(label)

            topScores, indexes = torch.topk(torch.tensor(binaryScores), topInBinary)
            hit = 0
            for index in indexes:
                if self._isPositivePair(binaryLabels[index]):
                    hit += 1
            topKPrecision = hit / topInBinary

            self.remoteLogger.experiment.log_metric(
                f"top-{topInEmbedding}-{topInBinary}-precision-cross",
                topKPrecision,
                step=self.batchStep,
                epoch=self.batchStep,
            )

        # Nearest top-K/L strategy
        # TOP-L/K
        embeddingLinkDict = {}

        for label in embeddingScoresOutput:
            monomerA, monomerB = label.split(':')
            score = embeddingScoresOutput[label]
            if monomerA not in embeddingLinkDict:
                embeddingLinkDict.update({monomerA: {}})
            if monomerB not in embeddingLinkDict:
                embeddingLinkDict.update({monomerB: {}})
            embeddingLinkDict[monomerA].update({monomerB: score})
            embeddingLinkDict[monomerB].update({monomerA: score})

        for topkl in self.nearestKStrategyRating:
            topEmbedding, topBinary = topkl
            print(f"top emb{topEmbedding}& top bin{topBinary}")

            targetLabels = []
            targetPositiveLabels = []

            targetBinaryLabels = []
            targetBinaryPositiveLabels = []

            totalHitsByEmbedding = 0
            totalHitsByBinary = 0
            for monomer in embeddingLinkDict:
                monomerLabel = []
                monomerScore = []
                for partner in embeddingLinkDict[monomer]:
                    if partner == monomer:
                        continue
                    if f"{monomer}:{partner}" in self.scoresOutputs:
                        monomerLabel.append(f"{monomer}:{partner}")
                        monomerScore.append(embeddingScoresOutput[f"{monomer}:{partner}"])
                    elif f"{partner}:{monomer}" in self.scoresOutputs:
                        monomerLabel.append(f"{partner}:{monomer}")
                        monomerScore.append(embeddingScoresOutput[f"{partner}:{monomer}"])
                    else:
                        print(f"{monomer}:{partner}")
                        raise ValueError

                binaryScoreForEmbeddingGuided = []
                binaryLabelForEmbeddingGuided = []

                topScores, indexes = torch.topk(torch.tensor(monomerScore), topEmbedding)
                hit = 0
                for index in indexes:
                    if self._isPositivePair(monomerLabel[index]):
                        hit += 1
                        if monomerLabel[index] not in targetPositiveLabels:
                            targetPositiveLabels.append(monomerLabel[index])
                    if monomerLabel[index] not in targetLabels:
                        targetLabels.append(monomerLabel[index])
                    binaryLabelForEmbeddingGuided.append(monomerLabel[index])
                    binaryScoreForEmbeddingGuided.append(self.scoresOutputs[monomerLabel[index]][0][0])
                if hit != 0:
                    totalHitsByEmbedding += 1

                topScores, indexes = torch.topk(torch.tensor(binaryScoreForEmbeddingGuided), topBinary)
                hit = 0
                for index in indexes:
                    if self._isPositivePair(binaryLabelForEmbeddingGuided[index]):
                        hit += 1
                        if binaryLabelForEmbeddingGuided[index] not in targetBinaryPositiveLabels:
                            targetBinaryPositiveLabels.append(binaryLabelForEmbeddingGuided[index])
                    if binaryLabelForEmbeddingGuided[index] not in targetBinaryLabels:
                        targetBinaryLabels.append(binaryLabelForEmbeddingGuided[index])
                if hit != 0:
                    totalHitsByBinary += 1

            self.remoteLogger.experiment.log_metric(
                f"Nearest-{topEmbedding}-Monomer-Recall",
                totalHitsByEmbedding,
                step=self.batchStep,
                epoch=self.batchStep,
            )

            self.remoteLogger.experiment.log_metric(
                f"NearestBinary-{topBinary}of{topEmbedding}-Monomer-Recall",
                totalHitsByBinary,
                step=self.batchStep,
                epoch=self.batchStep,
            )

            self.remoteLogger.experiment.log_metric(
                f"Nearest-{topEmbedding}-Pair-Recall",
                len(targetPositiveLabels),
                step=self.batchStep,
                epoch=self.batchStep,
            )

            self.remoteLogger.experiment.log_metric(
                f"Nearest-{topEmbedding}-Pair-Full",
                len(targetLabels),
                step=self.batchStep,
                epoch=self.batchStep,
            )

            self.remoteLogger.experiment.log_metric(
                f"NearestBinary-{topBinary}of{topEmbedding}-Pair-Full",
                len(targetBinaryLabels),
                step=self.batchStep,
                epoch=self.batchStep,
            )

            self.remoteLogger.experiment.log_metric(
                f"NearestBinary-{topBinary}of{topEmbedding}-Pair-Recall",
                len(targetBinaryPositiveLabels),
                step=self.batchStep,
                epoch=self.batchStep,
            )

            with open(f"comet_partners_tmp_{topEmbedding}", 'wb') as file:
                file.write(orjson.dumps(targetLabels))

            # Upload files
            self.remoteLogger.experiment.log_asset(
                file_data=f"comet_partners_tmp_{topEmbedding}",
                file_name=f"{self.modelName}-{self.batchStep}-KNNpartners-{topEmbedding}.json",
            )









