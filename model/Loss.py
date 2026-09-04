import torch
from torch import nn, Tensor

class InterfaceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.crossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, predictedInterface, groundTruthInterface):
        # print(f"Shape of predictedInterface: {predictedInterface.shape}")
        # print(f"Shape of groundTruthInterface: {groundTruthInterface.shape}")
        mask = torch.where(groundTruthInterface[:, 0] != -1)
        groundTruthInterface = groundTruthInterface.float()
        predictedInterface = predictedInterface[0, mask[0], :]
        groundTruthInterface = groundTruthInterface[mask[0], 1:]
        # print(f"Shape of predictedInterface(Processed): {predictedInterface.shape}")
        # print(f"Shape of groundTruthInterface(Processed): {groundTruthInterface.shape}")
        # print("Achieving ce loss")
        loss = self.crossEntropyLoss(predictedInterface, groundTruthInterface)
        # print("Finished")
        return loss


class PPIContrastiveLoss(nn.Module):

    def __init__(self, temperature = 1.0):
        super().__init__()
        print("Use sup. contrastive loss")
        self.temperature = temperature
        self.cosSimilarity = nn.CosineSimilarity(-1)

    def forward(self, positivePairMonomerA, positivePairMonomerB, negativeMonomers: list[Tensor]):
        positiveGroupA = torch.cat(positivePairMonomerA, dim=-2)
        positiveGroupB = torch.cat(positivePairMonomerB, dim=-2)
        positiveCosineSimilarity = self.cosSimilarity(positiveGroupA, positiveGroupB)
        negatives = torch.cat(negativeMonomers, dim=-2)

        negatives = negatives.unsqueeze(1).repeat(1, positiveGroupA.shape[0], 1)
        positiveGroupA = positiveGroupA.unsqueeze(0).repeat(negatives.shape[0], 1, 1)
        positiveGroupB = positiveGroupB.unsqueeze(0).repeat(negatives.shape[0], 1, 1)

        negativeCosineSimilarityA = self.cosSimilarity(positiveGroupA, negatives)
        negativeCosineSimilarityB = self.cosSimilarity(positiveGroupB, negatives)
        # temp
        positiveSimilarity = positiveCosineSimilarity / self.temperature
        negativeSimilarityA = negativeCosineSimilarityA / self.temperature
        negativeSimilarityB = negativeCosineSimilarityB / self.temperature
        # exp
        positiveSimilarity = torch.exp(positiveSimilarity)
        negativeSimilarityA = torch.exp(negativeSimilarityA)
        negativeSimilarityB = torch.exp(negativeSimilarityB)
        # neg sum
        negativeSimilaritySum = negativeSimilarityB.sum() + negativeSimilarityA.sum()
        factor = positiveSimilarity * torch.pow(negativeSimilaritySum + positiveSimilarity, -1)
        return (
            - torch.log(factor).sum() / factor.numel(),
            positiveCosineSimilarity.mean(),
            (negativeCosineSimilarityA + negativeCosineSimilarityB).mean() / 2.0
        )


class PPISigLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print("Use SigLIP loss")
        self.learnableNormConstantTemperature = nn.Parameter(torch.tensor([10.0]))
        self.cosSimilarity = nn.CosineSimilarity(-1)
        self.learnableNormConstantB = nn.Parameter(torch.Tensor([-10.0]))

    def forward(self, positivePairMonomerA, positivePairMonomerB, negativeMonomers: list[Tensor]):
        self.learnableNormConstantTemperature.data = torch.clamp(self.learnableNormConstantTemperature.data, min=0.0, max=16.0)
        positiveGroupA = torch.cat(positivePairMonomerA, dim=-2)
        positiveGroupB = torch.cat(positivePairMonomerB, dim=-2)
        positiveCosineSimilarity = self.cosSimilarity(positiveGroupA, positiveGroupB)
        # print(f"Positve cosine:{positiveCosineSimilarity}")
        negatives = torch.cat(negativeMonomers, dim=-2)
        negatives = negatives.unsqueeze(1).repeat(1, positiveGroupA.shape[0], 1)
        positiveGroupA = positiveGroupA.unsqueeze(0).repeat(negatives.shape[0], 1, 1)
        positiveGroupB = positiveGroupB.unsqueeze(0).repeat(negatives.shape[0], 1, 1)
        negativeCosineSimilarityA = self.cosSimilarity(positiveGroupA, negatives)
        negativeCosineSimilarityB = self.cosSimilarity(positiveGroupB, negatives)
        # print(f"Negative cosine A:{negativeCosineSimilarityA}")
        # print(f"Negative cosine B:{negativeCosineSimilarityB}")
        # Got exp factor
        positiveSimilarity = positiveCosineSimilarity * self.learnableNormConstantTemperature + self.learnableNormConstantB
        negativeSimilarityA = negativeCosineSimilarityA * self.learnableNormConstantTemperature + self.learnableNormConstantB
        negativeSimilarityB = negativeCosineSimilarityB * self.learnableNormConstantTemperature + self.learnableNormConstantB
        # print(f"With T:{self.learnableNormConstantTemperature.item()} and b:{self.learnableNormConstantB.item()}")
        # print(f"Positive Logits:{positiveSimilarity}")
        # print(f"Negative Logits A:{negativeSimilarityA}")
        # print(f"Negative Logits B:{negativeSimilarityB}")
        # exp
        positiveSimilarity = torch.nn.functional.logsigmoid(+positiveSimilarity)
        negativeSimilarityA = torch.nn.functional.logsigmoid(-negativeSimilarityA)
        negativeSimilarityB = torch.nn.functional.logsigmoid(-negativeSimilarityB)
        # print(f"Positive Contribution:{positiveSimilarity}")
        # print(f"Negative Contribution A:{negativeSimilarityA}")
        # print(f"Negative Contribution B:{negativeSimilarityB}")
        normalizedFactor = positiveCosineSimilarity.numel()
        # neg sum
        loss = (positiveSimilarity.sum() + negativeSimilarityA.sum() + negativeSimilarityB.sum()) / normalizedFactor
        return - loss, positiveCosineSimilarity.mean(), (negativeCosineSimilarityA + negativeCosineSimilarityB).mean() / 2.0
