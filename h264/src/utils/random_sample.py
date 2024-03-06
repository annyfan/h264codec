
import os
import random
import csv
from pathlib import Path
from h264.src.stages.utils import TrainStage

def random_sample(configPath):

    
        
    data = {
        
        "traindata": [],
        "testdata": [],
        "valdata": [],
    }

    index = 0

    with open(Path(configPath) / "dataset.csv") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        for row in reader:
            trainFlag = row[1]
            if trainFlag == TrainStage.TRAIN.name:
                data["traindata"].append(row[0])
            elif trainFlag == TrainStage.VALIDATE.name:
                data["valdata"].append(row[0])
            elif trainFlag == TrainStage.TEST.name:
                data["testdata"].append(row[0])
            else:
                raise ValueError("Invalid training flag.")
            index += 1

    trainlist = random.sample(data["traindata"], 50)
    validlist = random.sample(data["valdata"], 50)
    if os.path.exists(configPath + 'dataset_small.csv'):
        os.remove(configPath + 'dataset_small.csv')
    with open(Path(configPath) /  "dataset_small.csv",  mode='w') as f:

        writer = csv.writer(f)
        writer.writerow(['id','flag'])
        for row in trainlist:
            writer.writerow([row,'TRAIN'])
        for row in validlist:
            writer.writerow([row,'VALIDATE'])



if __name__ == "__main__":

    random_sample('/data/h264_v20231127')
