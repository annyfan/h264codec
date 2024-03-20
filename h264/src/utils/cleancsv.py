
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
        h264Path = Path(configPath) / "h264"

        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        for row in reader:
            trainFlag = row[1]
            if trainFlag == TrainStage.VALIDATE.name:
                if os.path.exists(str(h264Path / (row[0] + ".h264") )):

                    data["valdata"].append(row[0])
           

    with open(Path(configPath) /  "new_dataset.csv",  mode='w') as f:

        writer = csv.writer(f)
        for row in data["valdata"]:
            writer.writerow([row,'VALIDATE'])



if __name__ == "__main__":

    random_sample('C:\\tmp\\dataset\\h264_v20231127\\')
