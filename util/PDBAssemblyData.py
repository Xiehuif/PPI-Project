from Bio.PDB import MMCIFParser


def ReadCIFStructureFile(fileDir: str):
    structure = MMCIFParser(QUIET=True).get_structure(structure_id='NullID', filename=fileDir)
    return structure.child_list[0].child_list

def GetChainStructureFromCIF(cifStructures, chainID):
    for model in cifStructures:
        if model.id == chainID:
            return model
    return None
