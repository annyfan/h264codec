from os import walk
import os

from h264.src.utils.mp4parser.mp4parser import MP4
import os
import uuid
import re
import subprocess
import csv
import numpy as np


path = '/data/h264_v20230930/20230116_Valorant_tenz.mp4'
dirpath = '/data/h264_v20231124/'
resolution = '64x64'
h264dir = 'h264/'
imagedir = 'image/'

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

    uid = str(uuid.uuid4())
    #uid = '55ac670d-9a64-411c-8254-05238c62836f'

    # convert to lower resolution
    #ffmpeg -i 20230114_ApexLegends_faide.mp4 -vf "scale=64x64" 20230114_ApexLegends_faide_64x64.mp4
    if resolution is not None:
        try:
            newpath = path[:-4] + resolution + ".mp4"
            print(f'begin converting mp4:{newpath}')
            print('output',subprocess.check_output(['ffmpeg', '-i', path,   '-vf', 'scale=' + resolution, newpath]))
            path = newpath
            print('end converting mp4')
        except subprocess.CalledProcessError as err:
            print(err)    

    #-frame_pts true: use the frame index for image names, otherwise, the index starts from 1 and increments 1 each time
    #cmd = f'ffmpeg -skip_frame nokey -i {path}  -vsync vfr -frame_pts true {imagepath}{uid}-%d.jpeg'
    #os.system(cmd)
    try:
        print('begin extracting images')
        print('output:',subprocess.check_output(['ffmpeg', '-skip_frame', 'nokey','-i', path,  '-vsync' ,'vfr', '-frame_pts', 'true', imagepath + uid+'-%d.jpeg']))
        print('end extracting images')
    except subprocess.CalledProcessError as err:
        print(err)


    # get i frame index and extract frame from mp4
    iframe_index = []
    image_files = next(walk(imagepath), (None, None, []))[2]
    for frame_file in image_files:
        if frame_file.startswith(uid):
            m = re.search(uid + '-' + '(\d+).jpeg', frame_file)
            frame_idx = m.group(1)
            iframe_index.append(int(frame_idx))
            id_list.append(uid + '-' + str(frame_idx))


    #write frame into h264 path
    one_based = 1 if 0 not in iframe_index else 0

    mp4 = MP4(path)
    mp4.parse()
    frames = mp4.get_samples()
    iframe_files = []
    if frames is not None and len(frames):
        print('begin generate h264')
        iframe_files = mp4.write_vcl_nal_frame(path, [frames[i - one_based]  for i in iframe_index],
                                              [h264path + uid + '-' + str(j) + '.h264' for j in iframe_index])
        print('end generate h264')


list_len = len(id_list)
print(f'{list_len} h264s generated')

np.random.shuffle(id_list)
trainlist, validlist, testlist = np.split(id_list, 
                       [int(.8*list_len), int(.9*list_len)])

if os.path.exists('dataset.csv'):
    print('remove dataset.csv')
    os.remove('dataset.csv')

print('begin writing dataset.csv')
with open(dirpath + "dataset.csv",  mode='w+') as f:
        writer = csv.writer(f)
        writer.writerow(['id','flag'])
        for row in trainlist:
            writer.writerow([row,'TRAIN'])
        for row in validlist:
            writer.writerow([row,'VALIDATE'])
        for row in testlist:
            writer.writerow([row,'TEST'])

print('end writing dataset.csv')