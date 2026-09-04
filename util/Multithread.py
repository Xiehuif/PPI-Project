from multiprocessing import Process

import wget


def DownloadFile(url, outputDir, maxRetry):
    failFlag = 0
    while failFlag < maxRetry:
        try:
            wget.download(url, outputDir)
            return
        except:
            failFlag += 1
            continue
    print(f"Fail in download {url}")
    return



class MultithreadPool:

    def __init__(self, workerNumber):
        self.processNum = workerNumber
        self.processes: list[Process] = []

    def Run(self, target, args):
        while len(self.processes) == self.processNum:
            recycle = []
            for process in self.processes:
                if not process.is_alive():
                    recycle.append(process)
            for deadProcess in recycle:
                self.processes.remove(deadProcess)

        newProcess = Process(target=target, args=args)
        self.processes.append(newProcess)
        newProcess.start()

    def Sync(self):

        while len(self.processes) != 0:
            recycle = []
            for process in self.processes:
                if not process.is_alive():
                    recycle.append(process)
            for deadProcess in recycle:
                self.processes.remove(deadProcess)