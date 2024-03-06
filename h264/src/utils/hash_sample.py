
import hashlib
import csv
import os
from pathlib import Path
import torch
import torchvision
from h264.src.models.utils import split_batch

from h264.src.utils.mp4parser.mp4parser import MP4

def hash_sample(configPath):
 
    basePath  = Path(configPath)
    with open(basePath / "dataset.csv") as f, open(basePath / "dataset-file-hash-sha265.csv", 'w') as b:
        reader = csv.reader(f)
        writer = csv.writer(b)
        writer.writerow(['id','hash'])
        next(reader, None)  # skip the headers
        for row in reader:
            id = row[0]
                    # Load h264
            h264Filename = (
                    basePath
                    / "h264bak"
                    / f"{id}.h264"
            )
            out_h264 =  (
                    basePath
                    / "h264"
                    / f"{id}.h264"
            )

            MP4.convert_h264_data(str(h264Filename), str(out_h264))
            md5str =  hashlib.sha256(open(str(out_h264),'rb').read()).hexdigest()
            
            
            writer.writerow([id,md5str])

   

def hash_sample2(configPath):
 
    basePath  = Path(configPath)

    with open(basePath / "dataset-file-hash-sha265.csv", 'w') as b:
        writer = csv.writer(b)
        writer.writerow(['id','hash'])
        basePath  = Path(basePath)
        for filename in  os.listdir(basePath / "h264bak" ):
            
            id = filename.removesuffix(".h264")
                    # Load h264
            h264Filename = (
                basePath
                / "h264bak"
                / filename
            )
            out_h264 =  (
                basePath
                / "h264"
                / filename
            )
            
            if not os.path.exists(out_h264):
                MP4.convert_h264_data(str(h264Filename), str(out_h264))

            if os.path.exists(out_h264):  
                md5str =  hashlib.sha256(open(str(out_h264),'rb').read()).hexdigest()
            
            
            writer.writerow([id,md5str])

   



if __name__ == "__main__":

    hash_sample2('/data/h264_v20240206_7')

