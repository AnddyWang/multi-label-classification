# -*- coding: utf-8 -*-
#written by pengkai.wang
import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='/data0/users/pengkai1/datasets/MultiLabelImage',help='data directory')
    args = parser.parse_args()

    annonation_dirs=['动漫','家装家居','汽车','美食']
    for annonation_dir in annonation_dirs:
        for dirpath, subdirpath, filelist in os.walk(os.path.join(args.data_dir, annonation_dir)):
            for index, value in enumerate(filelist):
                if value.endswith('.jpg'):
                    filename = os.path.splitext(value)[0] + '.txt'
                    label_file = open(os.path.join(dirpath, filename), 'w')
                    print('{}={}'.format(annonation_dir, os.path.join(dirpath, filename)))
                    if annonation_dir == '动漫':
                        label_file.writelines('动漫')
                    elif annonation_dir == '家装家居':
                        label_file.writelines('家装家居')
                    elif annonation_dir == '汽车':
                        label_file.writelines('汽车')
                    elif annonation_dir == '美食':
                        label_file.writelines('美食')