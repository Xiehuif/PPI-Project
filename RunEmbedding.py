import argparse
from tqdm import tqdm
from util.Multithread import MultithreadPool
from util.ESMServerUtil import RunEmbeddings
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(description=f"Embedding entrance for protein monomers")
    parser.add_argument(
        "--apiKey",
        required=True,
        type=str,
        help="The API keys you applied from Biohub.ai to run ESM models",
    )
    parser.add_argument(
        "--inputSequences",
        required=True,
        metavar="/path/to/inputSequences.csv",
        type=str,
        help="Specify the sequence CSV file path, contains header \'proteinID\' & \'seq\', "
             "where each row corresponding to the monomer id you would like to specified and monomers' sequence",
    )
    parser.add_argument(
        "--output",
        metavar="/path/to/outputDir",
        required=True,
        type=str,
        help="Specify the embedding file output path, must be a directory. If the output directory does not exist, it will be created.",
    )
    parser.add_argument(
        '--processNumber',
        default=10,
        type=int,
        metavar='10',
        help="Specify the embedding process number, default is 16, which achieve a good performance-load balance "
             "for our machine. You should setting this argument to adapt to your machine specs for better performance.",
    )
    parser.add_argument(
        '--batchSize',
        default=10,
        type=int,
        metavar='10',
        help="Specify the batch size for each post action, default is 10. A larger batch size reduces overhead "
             "by sending fewer network requests. A smaller batch size is more resilient to unstable connections "
             "and prevents 'payload too large' errors.",
    )

    print(f"Loading data from {parser.parse_args().inputSequences}")
    inputData = pd.read_csv(parser.parse_args().inputSequences, na_filter=False)
    sequenceData = {}
    length = len(inputData['proteinID'])
    for i in tqdm(range(length)):
        proteinID = str(inputData['proteinID'][i])
        sequence = str(inputData['sequence'][i])
        if proteinID in sequenceData:
            print(f"WARNING: ID:{proteinID} has maps to multiple sequences\nalready reads:"
                  f"\n{sequenceData[proteinID]}\n"
                  f"yet another reads:"
                  f"\n{sequence}\n"
                  f"the last sequence would replace previously mapped sequences."
            )
        sequenceData.update({proteinID: sequence})
    fullSequenceNumber = len(sequenceData)
    print(f"Found sequence number: {fullSequenceNumber}")
    # User specified info
    outputDir = parser.parse_args().output
    batchSize = parser.parse_args().batchSize
    threadPool = MultithreadPool(parser.parse_args().processNumber)
    apiToken = parser.parse_args().apiKey

    # Round info
    outputFolder = ['ESM3', 'ESMC']
    modelEndpoints = ["esm3-large-2024-03", "esmc-6b-2024-12"]

    # Program data
    for i in range(2):
        outputEmbeddingDir = f"{outputDir}/{outputFolder[i]}"
        os.makedirs(outputEmbeddingDir, exist_ok=True)
        taskNumber = 0
        batchName = []
        batchSequence = []
        modelEndpoint = modelEndpoints[i]
        print(f"Running the {outputFolder[i]} Embeddings")
        for proteinID in tqdm(sequenceData):
            taskNumber += 1
            sequence = sequenceData[proteinID]
            fileName = f"{outputEmbeddingDir}/{proteinID}"
            if os.path.exists(fileName):
                continue
            batchName.append(fileName)
            batchSequence.append(sequence)
            if len(batchSequence) == batchSize and len(batchName) == batchSize:
                threadPool.Run(RunEmbeddings, (batchSequence, batchName, apiToken, 5, modelEndpoint))
                batchName = []
                batchSequence = []
        threadPool.Run(RunEmbeddings, (batchSequence, batchName, apiToken, 5, modelEndpoint))
        threadPool.Sync()

    print(f"Saving the CSV file as inference input...")
    csvOutput = {
        'proteinID': [],
        'esm3EmbeddingFile': [],
        'esmcEmbeddingFile': [],
    }
    for proteinID in tqdm(sequenceData):
        esm3FileDir = f"{outputDir}/{outputFolder[0]}/{proteinID}"
        esmcFileDir = f"{outputDir}/{outputFolder[1]}/{proteinID}"
        if os.path.exists(esmcFileDir) and os.path.exists(esm3FileDir):
            csvOutput['proteinID'].append(proteinID)
            csvOutput['esm3EmbeddingFile'].append(esm3FileDir)
            csvOutput['esmcEmbeddingFile'].append(esmcFileDir)
    pd.DataFrame(csvOutput).to_csv(f'{outputDir}/MonomerEmbeddingFiles.csv', index=False)

if __name__ == '__main__':
    main()