import os
import yaml
import cv2
import numpy as np

def load_params():
    '''
    Loads params from config.yaml
    '''
    src_dir = os.path.dirname(os.path.realpath(__file__))
    rel_path = '/'
    filename = 'config.yaml'
    with open(src_dir+rel_path+filename, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def decompress(line, params):
    '''
    Decompresses line points into an image
    '''
    img = np.zeros((params['img_length'], params['img_width']))
    for idx in range(0,len(line)):
        if idx != 0:
            prev_point = line[idx-1]
            point = line[idx]
            if prev_point[0] == point[0]:
                cv2.line(img, (point[0], min(prev_point[1], point[1])), (point[0], max(prev_point[1],point[1])), [1], params['thickness'])
            else:
                cv2.line(img, (min(prev_point[0], point[0]), point[1]), (max(prev_point[0],point[0]), point[1]), [1], params['thickness'])
    return img

if __name__ == '__main__':
    params = load_params()
    test = np.load('./task_dataset/dataset.npy', allow_pickle=True)
    img = decompress(test[0], params)
    import ipdb; ipdb.set_trace()
