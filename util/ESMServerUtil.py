from typing import Literal

from esm.sdk.forge import ESM3ForgeInferenceClient, ESMCForgeInferenceClient
from esm.sdk import batch_executor
from esm.sdk.api import ESMProteinError, LogitsOutput, LogitsConfig, ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from torch import Tensor, save


def RunBatch(
        sequenceBatch,
        apikey,
        model
):
    def embed_sequence(client: ESM3ForgeInferenceClient, sequence: str) -> LogitsOutput:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        if isinstance(protein_tensor, ESMProteinError):
            raise protein_tensor
        output = client.logits(protein_tensor, LogitsConfig(return_embeddings=True))
        return output

    # print('finish define')
    client = ESM3ForgeInferenceClient(model=model, url="https://biohub.ai",
                                      token=apikey)

    # Usage Example:
    # To execute a batch job, wrap your function inside the batch executor context manager.
    # Syntax:
    # with batch_executor() as executor:
    #     outputs = executor.execute_batch(user_func=<your_function>, **kwargs)

    with batch_executor(show_progress=False) as executor:
        outputs = executor.execute_batch(user_func=embed_sequence, client=client, sequence=sequenceBatch)

    result = []
    for output in outputs:
        result.append(output.embeddings)
    return result


def RunEmbeddings(batch, outputDirs, apikey, patience=5, embeddingModel: Literal["esm3-large-2024-03", "esmc-6b-2024-12"] | None=None):
    if len(batch) == 0:
        return
    if len(batch) != len(outputDirs):
        raise ValueError('Length of sequence batch and output directory list do not match')
    # print('Prepare ESMC Embs')
    embeds = None
    for i in range(patience):
        try:
            embeds: Tensor = RunBatch(batch, apikey, embeddingModel)
            break
        except:
            print(f"An error occurs, retry, patience{patience}, now{i}")
            continue
    if embeds is None:
        print('Skip bad request... You may retry later.')
        return
    # print('Saving...')
    for i in range(len(batch)):
        save(embeds[i], outputDirs[i])

