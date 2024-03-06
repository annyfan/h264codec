
import os
from pathlib import Path

from h264.src.utils.mp4parser.mp4parser import MP4



   

def cleanup_samples(configPath):
    clean_folder = "h264clean"
    basePath  = Path(configPath)

    if not os.path.exists(basePath / clean_folder ):
        os.makedirs(os.path.join(basePath, clean_folder))
        

    for filename in  os.listdir(basePath / "h264" ):
        
                    # Load h264
        h264Filename = (
            basePath
            / "h264"
            / filename
        )
        out_h264 =  (
            basePath
            / clean_folder
            / filename
        )
            
        if not os.path.exists(out_h264):
            MP4.convert_h264_data(str(h264Filename), str(out_h264))

           

   



if __name__ == "__main__":
    cleanup_samples('/data/dataset/h264_v20231127')
    cleanup_samples('/data/dataset/h264_v20240206_1')
    cleanup_samples('/data/dataset/h264_v20240206_2')
    cleanup_samples('/data/dataset/h264_v20240206_3')
    cleanup_samples('/data/dataset/h264_v20240206_4')
    cleanup_samples('/data/dataset/h264_v20240206_5')
    cleanup_samples('/data/dataset/h264_v20240206_6')
    cleanup_samples('/data/dataset/h264_v20240206_7')
