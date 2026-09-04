import os
import modeller
import modeller.automodel


reference = {
    'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'PRO': 'P', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W',
    'SER': 'S', 'THR': 'T', 'CYS': 'C', 'MET': 'M', 'ASN': 'N', 'GLN': 'Q', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
    'ARG': 'R', 'HIS': 'H', 'UNK': 'X'
}

residueNames = []
for threeLetterWord in reference:
    if reference[threeLetterWord] not in residueNames:
        residueNames.append(reference[threeLetterWord])


def GetModellerEnvironment():
    env = modeller.Environ()
    env.io.hetatm = True  # type:ignore
    return env

def MakeAlignment(modellerEnvironment, structureFile: str, sequence: str, proteinID: str, chainID: str, outputDir: str):
    structureModel = modeller.Model(
        modellerEnvironment,
        file=structureFile,
        model_format='MMCIF',
        model_segment=(f'FIRST:{chainID}', f'LAST:{chainID}')
    )
    alignmentCode = f"{proteinID}-{chainID}"
    aln = modeller.Alignment(modellerEnvironment)
    errorSequence = sequence
    newSequence = ''
    for word in errorSequence:
        if word not in residueNames:
            newSequence = newSequence + '.'
        else:
            newSequence = newSequence + word
    aln.append_sequence(newSequence)
    aln[0].code = alignmentCode
    aln.append_model(mdl=structureModel, align_codes="Structure", atom_files=structureFile)
    aln.salign()
    aln['Structure'].name = f'{proteinID}:{chainID}'
    aln.write(file=f'{outputDir}', alignment_format='PIR')

def _ReadAlignment(dir):
    with open(dir, 'r') as file:
        content = file.read()
        content = content.split('\n')
    readingFlag = False
    headFlag = False
    targetKey = None
    output = {}
    for line in content:
        if line == '':
            continue
        if '>' in line:
            readingFlag = True
            headFlag = True
            continue
        if readingFlag and headFlag:
            if 'sequence' in line:
                targetKey = 'sequence'
            else:
                targetKey = 'structure'
            output.update({targetKey: ''})
            headFlag = False
            continue
        if readingFlag:
            if '*' not in line:
                output[targetKey] = output[targetKey] + line
            else:
                newLine = line[:-1]
                output[targetKey] = output[targetKey] + newLine
                readingFlag = False
    return output


def ReadAlignment(structureFile: str, sequence: str, proteinID: str, chainID: str, outputDir: str):
    filePath = f"{outputDir}/{proteinID}.pir"
    if os.path.exists(filePath):
        print(f"Using cached:{filePath}")
        return _ReadAlignment(filePath)
    else:
        print(f"Making aln:{filePath}")
        os.makedirs(outputDir, exist_ok=True)
        env = GetModellerEnvironment()
        MakeAlignment(env, structureFile, sequence, proteinID, chainID, filePath)
        return _ReadAlignment(filePath)

