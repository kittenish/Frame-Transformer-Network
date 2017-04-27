import numpy as np
import os

#f = open('/Users/mac/Desktop/train_theta_result_541.txt','r')
def cal_position(filepath,newpath):
    f = open(filepath,'r')
    theta_1 = 0.0
    theta_2 = 0.0
    position = np.zeros([461, 2])
    #position = np.zeros([100, 2])
    i = 0

    for i in range(461):
    #for i in range(100):
        line = f.readline()

        if line:
            p=line.rfind(' ')
            theta_0 = float(line[0:p])
            theta_1 = float(line[p:])
            # position[i][1] = 15 * (theta_0 + theta_1 + 1)
            # position[i][0] = 15 * (theta_1 + 1 - theta_0)
            position[i][1] = 100 * (theta_0 + theta_1 + 1)
            position[i][0] = 100 * (theta_1 + 1 - theta_0)
            temp = 0
            if position[i][0] > position[i][1]:
                temp = position[i][1]
                position[i][1] = position[i][0]
                position[i][0] = temp
            if position[i][0] < 0:
                position[i][0] = 0
            if position[i][1] > 200:
                position[i][1] = 200
            i = i + 1

    np.savetxt(newpath,position,fmt="%d")
    f.close()

def dirlist(path):

    filelist =  os.listdir(path)
    #gt = '/Users/mac/Desktop/Ekman/5*1_initial=0.2/train/train_position.txt'

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if filename == 'train':
            pass
        elif os.path.isdir(filepath):
            dirlist(filepath)
        else:
            #allfile.append(filepath)
            if filename[-4:len(filename)] == '.txt' and filename != 'accuracy.txt' :

                newpath = '/'.join(filepath.split('/')[0:6]) + '/position/' + filename
                cal_position(filepath,newpath)

                print newpath

dirlist('/Users/mac/Desktop/Ekman/')

#cal_position('/Users/mac/Desktop/video_spacial/syhthetic/color/result/4096*3_initial=0.5/train/train_theta.txt','/Users/mac/Desktop/video_spacial/syhthetic/color/result/4096*3_initial=0.5/train/train_position.txt')
