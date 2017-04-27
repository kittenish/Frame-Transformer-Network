import numpy as np
import matplotlib.pyplot as plt
import os
import json

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def cal_mAP(url1, url2, alpha):
    f = open(url1, 'r')
    data = f.read()

    g = open(url2, 'r')
    gt = g.read()

    data = data.split('\n')
    data = data[0:len(data) - 1]
    pre = np.zeros((len(data), 2))

    num = 0
    for i in data:
        temp = i.split(' ')
        pre[num,0] = int(temp[0])
        pre[num,1] = int(temp[1])
        num = num + 1

    gt = gt.split('\n')
    gt = gt[0:len(gt) - 1]

    ground = np.zeros((len(gt), 2))

    num = 0
    for i in gt:
        temp = i.split(' ')
        ground[num,0] = int(temp[0])
        ground[num,1] = int(temp[1])
        num = num + 1

    tiou = np.zeros(len(data))
    tp = np.zeros(len(data))
    fp = np.zeros(len(data))
    
    b_i = np.maximum(pre[:,0], ground[:,0])
    e_i = np.minimum(pre[:,1], ground[:,1])

    b_u = np.minimum(pre[:,0], ground[:,0])
    e_u = np.maximum(pre[:,1], ground[:,1])

    tiou = (e_i - b_i) / (e_u - b_u)

    for i in range(len(tiou)):
        if tiou[i] < alpha:
            fp[i] = 1
        else:
            tp[i] = 1
    tp = np.cumsum(tp).astype(np.float)
    fp = np.cumsum(fp).astype(np.float)
    rec = tp / len(data)
    prec = tp / (tp + fp)
    return interpolated_prec_rec(prec, rec)

def cal_filter_mAP(url, alpha):
    ground_url = url + '/train/train_position.txt'
    filelist = os.listdir(url + '/position/')
    filelist = sorted(filelist)
    mAP = np.zeros(100)
    num = 0
    for filename in filelist :
        if filename[-4:len(filename)] == '.txt' and filename[0] == 't':
            print url + '/position/' + filename
            mAP[num] = cal_mAP(url + '/position/' + filename, ground_url, alpha)
            num = num + 1
    return mAP

def rename():
    path = '/Users/mac/Desktop/ff/'
    filelist =  os.listdir(path)
    for i in filelist:
        if i != '.DS_Store':
            url = path + i + '/position/'
            filelist_n = os.listdir(url)
            for j in filelist_n:
                if j != '.DS_Store':
                    name = j.split('_')
                    if len(name[3]) != '7':
                        if len(name[3]) == 6:
                            name[3] = '0' + name[3]
                            name = '_'.join(name)
                            os.rename(url + j, url + name)
                        if len(name[3]) == 5:
                            name[3] = '00' + name[3]
                            name = '_'.join(name)
                            os.rename(url + j, url + name)


def cal_all_mAP(alpha):
    path = '/Users/mac/Desktop/ff/'
    filelist =  os.listdir(path)
    all_mAP = {}
    for i in filelist:
        if i != '.DS_Store' and i[0] != 'z':
            url = path + i + '/position/'
            mAP = cal_filter_mAP(path + i, alpha)
            all_mAP[i] = list(mAP)

    ff = open('mAP.txt', 'w')
    ff.write(json.dumps(all_mAP))
    ff.close()

def draw_mAP():
    ff = open('/Users/mac/Desktop/mAP.txt', 'r')
    all_mAP = ff.read()
    all_mAP = json.loads(all_mAP)
    num = 0
    color = ['#00BFFF','#6495ED','darkorange','r']
    plt.figure(figsize=(10,5))
    keys = all_mAP.keys()
    keys = sorted(keys)
    for i in keys:
        
        plt.subplot(1,2,num+1)
        plt.title(str(i) + 'test' )
        x = np.arange(0,500,5)
        y = all_mAP[i]
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.plot(x,y)
        # mAP = all_mAP[i]
        # x = np.arange(1,1001,40)
        # y = np.zeros(25)
        # #print len(y)
        # for j in range(25):
        #     #print i*5
        #     y[j] = max(mAP[j*8:j*8+8])
        # #y = mAP
        # if i == '5*1':
        #     plt.plot(x,y, label = i)#, color=color[num], linewidth=2.5)
        # else:    
        #     plt.plot(x,y,label=i)#, color=color[num])
        num = num + 1
    plt.legend(loc='lower right')
    plt.show()

alpha = 0.7

if __name__=='__main__':
    #rename()
    #cal_all_mAP(alpha)
    draw_mAP()
