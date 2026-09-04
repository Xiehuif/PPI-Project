import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from util.PDBAssemblyData import GetChainStructureFromCIF, ReadCIFStructureFile
from util.ModellerUtil import reference, ReadAlignment
import argparse


thresholdForInterfaces = [4.0, 8.0, 12.0]
binNumber = len(thresholdForInterfaces)

def MakeContactMap(inputPDBFile: str, authIDA, authIDB, sequenceAlignmentA, sequenceAlignmentB,
                   structureSequenceA, structureSequenceB):
    # Init
    sizeA = len(sequenceAlignmentA)
    sizeB = len(sequenceAlignmentB)
    distanceMap = torch.zeros(sizeA, sizeB)
    distanceMap[:, :] = -1.0
    # reading structure
    cifStructures = ReadCIFStructureFile(inputPDBFile)
    structureA = GetChainStructureFromCIF(cifStructures, authIDA)
    structureB = GetChainStructureFromCIF(cifStructures, authIDB)

    # Iteratively mapping distance
    indexA = -1
    for residueA in structureA:
        residueName = residueA.resname
        if residueName not in reference:
            continue
        residueOneLetterWordA = reference[residueName]
        while True:
            indexA += 1
            # Over length
            if indexA == sizeA:
                break
            # Check if this is the residue we want
            if structureSequenceA[indexA] == residueOneLetterWordA:
                break
        # Over length
        if indexA == sizeA:
            break
        # Now check if this residue overlap with FASTA sequence
        if residueOneLetterWordA != sequenceAlignmentA[indexA]:
            continue
        # Now that we check this is the residue we need!
        # Go to B chain
        indexB = -1
        for residueB in structureB:
            residueName = residueB.resname
            if residueName not in reference:
                continue
            residueOneLetterWordB = reference[residueName]
            while True:
                indexB += 1
                # Over length
                if indexB == sizeB:
                    break
                # Check if this is the residue we want
                if structureSequenceB[indexB] == residueOneLetterWordB:
                    break
            # Over length
            if indexB == sizeB:
                break
            # Now check if this residue overlap with FASTA sequence
            if residueOneLetterWordB != sequenceAlignmentB[indexB]:
                continue
            # Now that we check this is the residue we need!
            # we have now locked the residue pair residueA & residueB for [indexA, indexB] on the map
            # calculate the distance
            atomDistancePairs = np.array([[atom1 - atom2 for atom2 in residueA if atom2.element != 'H']
                                for atom1 in residueB if atom1.element != 'H'])
            distance = np.min(atomDistancePairs)
            # Good, now the distance is what we want for this residue pair
            distanceMap[indexA, indexB] = distance.item()

    # Finished
    return distanceMap

def MakeInterfaces(distanceMap, outputDirA, outputDirB):

    # bin: -1 : Unclear --- 0-3: <4.0 ; 4.0-8.0 ; 8.0-12.0 ; > 12.0 ;
    interfacesA = torch.zeros([distanceMap.shape[0], binNumber + 2]).long()
    interfacesB = torch.zeros([distanceMap.shape[1], binNumber + 2]).long()

    # Iteratively check
    for i in range(distanceMap.shape[0]):
        distancesOfResidue = distanceMap[i,:]
        maxDistance = distancesOfResidue.max()
        # Unmodeled residue
        if maxDistance < 0.0:
            interfacesA[i][0] = -1
            continue
        # set bins
        distancesOfResidueMask = torch.where(distancesOfResidue == -1.0)
        distancesOfResidue[distancesOfResidueMask] = 999.0
        minDistance = distancesOfResidue.min()
        targetBin = binNumber
        minRange = 0.0
        for distanceIndex in range(len(thresholdForInterfaces)):
            maxRange = thresholdForInterfaces[distanceIndex]
            if minDistance > minRange and minDistance < maxRange:
                targetBin = distanceIndex
            minRange = maxRange
        interfacesA[i][targetBin + 1] = 1
        distancesOfResidue[distancesOfResidueMask] = -1.0

    for i in range(distanceMap.shape[1]):
        distancesOfResidue = distanceMap[:,i]
        maxDistance = distancesOfResidue.max()
        # print(i, maxDistance)
        # Unmodeled residue
        if maxDistance < 0.0:
            interfacesB[i][0] = -1
            continue
        # set bins
        distancesOfResidueMask = torch.where(distancesOfResidue == -1.0)
        distancesOfResidue[distancesOfResidueMask] = 999.0
        minDistance = distancesOfResidue.min()
        targetBin = binNumber
        minRange = 0.0
        maxRange = 0.0
        for distanceIndex in range(len(thresholdForInterfaces)):
            maxRange = thresholdForInterfaces[distanceIndex]
            if minDistance > minRange and minDistance < maxRange:
                targetBin = distanceIndex
            minRange = maxRange
        interfacesB[i][targetBin + 1] = 1
        distancesOfResidue[distancesOfResidueMask] = -1.0

    torch.save(interfacesA, outputDirA)
    torch.save(interfacesB, outputDirB)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"Demonstrate how we extract interface from a CIF file.")
    parser.add_argument(
        "--inputCSV",
        required=True,
        type=str,
        help="The CSV file that contains the information we need to extract interfaces.",
    )
    parser.add_argument(
        "--output",
        metavar="/path/to/outputDir",
        required=True,
        type=str,
        help="Specify the output path for the interfaces.",
    )
    fileDir = parser.parse_args().inputCSV
    outputDir = parser.parse_args().output
    print(f"Loading input file: {fileDir}")
    inputCSVData = pd.read_csv(fileDir, na_filter=False)
    length = len(inputCSVData['proteinIDA'])
    print(f"Extracting interfaces...")
    for i in tqdm(range(length)):
        proteinIDA = inputCSVData['proteinIDA'][i]
        cifFile = inputCSVData['structureFile'][i]
        sequenceA = inputCSVData['sequenceA'][i]
        chainIDA = inputCSVData['chainIDA'][i]
        sequenceB = inputCSVData['sequenceB'][i]
        proteinIDB = inputCSVData['proteinIDB'][i]
        chainIDB = inputCSVData['chainIDB'][i]

        # alignments
        alignmentDir = f"{outputDir}/alignments"
        os.makedirs(alignmentDir, exist_ok=True)
        alignmentA = ReadAlignment(cifFile, sequenceA, proteinIDA, chainIDA, alignmentDir)
        alignmentB = ReadAlignment(cifFile, sequenceB, proteinIDB, chainIDB, alignmentDir)

        sequenceAlignmentA = alignmentA['sequence']
        sequenceAlignmentB = alignmentB['sequence']
        structureAlignmentA = alignmentA['structure']
        structureAlignmentB = alignmentB['structure']

        # distance maps
        distanceMapDir = f"{outputDir}/distanceMaps"
        os.makedirs(distanceMapDir, exist_ok=True)
        contactMap = MakeContactMap(
            cifFile,
            chainIDA,
            chainIDB,
            sequenceAlignmentA,
            sequenceAlignmentB,
            structureAlignmentA,
            structureAlignmentB
        )
        outputFileDir = f"{distanceMapDir}/{proteinIDA}&{proteinIDB}"
        torch.save(contactMap, outputFileDir)

        # interfaces
        interfaceDir = f"{outputDir}/interfaces"
        os.makedirs(interfaceDir, exist_ok=True)
        outputDirA = f"{interfaceDir}/{proteinIDA}"
        outputDirB = f"{interfaceDir}/{proteinIDB}"
        MakeInterfaces(contactMap, outputDirA, outputDirB)

