from os import listdir, walk
import os
from h264.src.utils.mp4parser.mp4parser import MP4
import os
import uuid
import re
import subprocess
import csv
import numpy as np

# trace the h264
#ffmpeg -i 1.h264 -c copy -bsf:v trace_headers -f null - 2> NALUS.txt
# generate pic 
#ffmpeg -framerate 25 -i 70f695c7-ff74-4ff8-b5be-064a0e221ad8-99994.h264 -frames:v 1 output.png
#ffmpeg -debug mb_type -i 70f695c7-ff74-4ff8-b5be-064a0e221ad8-99994.h264

path = '/data/h264_v20231121/20230114_ApexLegends_faide64x64.mp4'
dirpath = '/data/h264_v20231123/'
h264dir = 'h264ffmpeg/'
imagedir = 'image/'
uid = '55ac670d-9a64-411c-8254-05238c62836f'

h264path = os.path.join(dirpath, h264dir)
isExist = os.path.exists(h264path)
if not isExist:
    os.makedirs(h264path)
imagepath = os.path.join(dirpath, imagedir)
isExist = os.path.exists(imagepath)
if not isExist:
    os.makedirs(imagepath)

dirpath = os.path.expanduser(dirpath)
json_list = []
id_list = []

if path.endswith('.mp4'):

     # 1based
    #ffmpeg -i 20230114_ApexLegends_faide_512.mp4  -f image2  -vcodec copy  -bsf h264_mp4toannexb "%d.h264" 
    cmd = f'ffmpeg -i {path} -f image2  -vcodec copy  -bsf h264_mp4toannexb {dirpath}"%d.h264"'
    os.system(cmd)
  
           
    iframe_index = []
    image_files = next(walk(imagepath), (None, None, []))[2]
    print(len(image_files))
    for frame_file in image_files:
        if frame_file.startswith(uid):
            m = re.search(uid + '-' + '(\d+).jpeg', frame_file)
            frame_idx = m.group(1)
            iframe_index.append(int(frame_idx))
            id_list.append( uid + '-' + str(frame_idx))



    iframe_files = []
    add_one_based = 1 if 0 in iframe_index else 0
    if iframe_index is not None and len(iframe_index):
        for j in iframe_index:
                    
            cmd = f'mv {dirpath}{j + add_one_based}.h264 {h264path}{uid}-{j}.h264'
            os.system(cmd)


list_len = len(id_list)
print(f'{list_len} h264s generated')

for filename in  listdir(dirpath):
    fpath = os.path.join(dirpath, filename)
    # print(path.encode('utf-8'))
    if fpath.endswith('.h264'):
        try:
            #print(fpath)
            os.remove(fpath)
        except OSError:
             pass
