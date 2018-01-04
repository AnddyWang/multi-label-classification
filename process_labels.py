#written by pengkai.wang
import os
import argparse
import numpy as np

def parser_label_file(image_label_dir,filename):
    result=''
    with open(os.path.join(image_label_dir,filename+'.txt'),'r') as f:
        lines=f.readlines()
        for line in lines:
            result += line.strip() + ' '

    return result

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_dir',type=str,default='./images',help='data directory')
    #parser.add_argument('--image_label_dir',type=str,default='./image_label_dir',help='image label dir')
    #parser.add_argument('--output_dir', type=str, default='./', help='output dir')
    parser.add_argument('--data_dir',type=str,default='/data0/users/pengkai1/datasets/MultiLabel/images',help='data directory')
    parser.add_argument('--image_label_dir',type=str,default='/data0/users/pengkai1/datasets/MultiLabel/image_label_dir',help='image label dir')
    parser.add_argument('--output_dir', type=str, default='/data0/users/pengkai1/datasets/MultiLabel', help='output dir')
    args = parser.parse_args();

    train_lists=open(os.path.join(args.output_dir,'train_lists.txt'),'w')
    train_labels = open(os.path.join(args.output_dir, 'train_labels.txt'), 'w')
    test_lists = open(os.path.join(args.output_dir, 'test_lists.txt'), 'w')
    test_labels = open(os.path.join(args.output_dir, 'test_labels.txt'), 'w')

    validation_index=np.random.permutation(26887)[:5377]
    #validation_index = np.random.permutation(10)[:5]
    for root,path,filelist in os.walk(args.data_dir):
        for index,value in enumerate(filelist):
            if index in validation_index:
                test_lists.write(value+'\n')
                test_labels.write(parser_label_file(args.image_label_dir,value)+'\n')
            else:
                train_lists.write(value+'\n')
                train_labels.write(parser_label_file(args.image_label_dir,value)+'\n')