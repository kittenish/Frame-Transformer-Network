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
            pre[num,0] = eval(temp[0])
            pre[num,1] = eval(temp[1])
            num = num + 1

        gt = gt.split('\n')
        gt = gt[0:len(gt) - 1]

        ground = np.zeros((len(gt)-2, 7))

        num = 0
        for i in gt:
            temp = i.split(' ')
            ground[num,0] = eval(temp[0]) 
            ground[num,1] = eval(temp[1]) 
            ground[num,2] = eval(temp[2]) 
            ground[num,3] = eval(temp[3]) 
            ground[num,4] = eval(temp[4]) 
            ground[num,5] = eval(temp[5]) 
            ground[num,6] = eval(temp[6])
            if ground[num,6] <= 10:
                pre[num,0] = pre[num,0] / 10
                pre[num,1] = pre[num,1] / 10
                if pre[num,1] > ground[num,6]:
                    pre[num,1] = ground[num,6]
            else:
                pre[num,0] = pre[num,0] * ground[num,6] / 100
                pre[num,1] = pre[num,1] * ground[num,6] / 100
            num = num + 1

        tiou = np.zeros(len(data))
        tp = np.zeros(len(data))
        fp = np.zeros(len(data))

        b_i_1 = np.maximum(pre[:,0], ground[:,0])
        e_i_1 = np.minimum(pre[:,1], ground[:,1])

        b_u_1 = np.minimum(pre[:,0], ground[:,0])
        e_u_1 = np.maximum(pre[:,1], ground[:,1])

        b_i_2 = np.maximum(pre[:,0], ground[:,2])
        e_i_2 = np.minimum(pre[:,1], ground[:,3])

        b_u_2 = np.minimum(pre[:,0], ground[:,2])
        e_u_2 = np.maximum(pre[:,1], ground[:,3])

        b_i_3 = np.maximum(pre[:,0], ground[:,4])
        e_i_3 = np.minimum(pre[:,1], ground[:,5])

        b_u_3 = np.minimum(pre[:,0], ground[:,4])
        e_u_3 = np.maximum(pre[:,1], ground[:,5])

        tiou_1 = (e_i_1 - b_i_1) / (e_u_1 - b_u_1)
        tiou_2 = (e_i_2 - b_i_2) / (e_u_2 - b_u_2)
        tiou_3 = (e_i_3 - b_i_3) / (e_u_3 - b_u_3)
        print 
        count  = 0
        for i in range(len(tiou)):
            
            if (tiou_1[i] < alpha and (tiou_2[i] < alpha or np.isnan(tiou_2[i])) and (tiou_3[i] < alpha or np.isnan(tiou_3[i]))) :
                fp[i] = 1

            else:
                tp[i] = 1
                
        tp = np.cumsum(tp).astype(np.float)
        fp = np.cumsum(fp).astype(np.float)
        rec = tp / len(data)
        prec = tp / (tp + fp)
        return interpolated_prec_rec(prec, rec)

def cal_filter_mAP(url, alpha):
    ground_url = url + '/train/position.txt'
    filelist = os.listdir(url + '/position/')
    filelist = sorted(filelist)
    mAP = np.zeros(200)
    num = 0
    for filename in filelist :
        if filename[-4:len(filename)] == '.txt' :
            print ground_url
            print url + '/position/' + filename
            mAP[num] = cal_mAP(url + '/position/' + filename, ground_url, alpha)
            num = num + 1
    return mAP


def rename():
    path = '/Users/mac/Desktop/Ekman_200/'
    filelist =  os.listdir(path)
    for i in filelist:
        if i != '.DS_Store':
            url = path + i + '/position/'
            filelist_n = os.listdir(url)
            for j in filelist_n:
                if j != '.DS_Store':
                    name = j.split('_')
                    if len(name[2]) != '7':
                        if len(name[2]) == 6:
                            name[2] = '0' + name[2]
                            name = '_'.join(name)
                            os.rename(url + j, url + name)
                        if len(name[2]) == 5:
                            name[2] = '00' + name[2]
                            name = '_'.join(name)
                            os.rename(url + j, url + name)


def cal_all_mAP():
    path = '/Users/mac/Desktop/video_spacial/Ekman_100/'
    filelist =  os.listdir(path)
    all_mAP = {}
    for i in filelist:
        if i != '.DS_Store' and i[0] != 'z' :
            
            url = path + i + '/position/'
            mAP = cal_filter_mAP(path + i)
            all_mAP[i] = list(mAP)

    ff = open('mAP_Ekman6.txt', 'w')
    ff.write(json.dumps(all_mAP))
    ff.close()

def draw_mAP():
    ff = open('mAP_Ekman6.txt', 'r')
    all_mAP = ff.read()
    all_mAP = json.loads(all_mAP)
    num = 1
    v = 0
    plt.figure(figsize=(20,10))
    for i in all_mAP.keys():
        plt.subplot(2,3,num)
        plt.title(i)
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        mAP = all_mAP[i]
        x = np.arange(0,200,1)
        y = mAP
        
        if int(i) != 3 and int(i) != 6:
            #print max(y)
            v = v + max(y)
    
        plt.plot(x,y)
        num = num + 1
    plt.show()
    print v / 4.0


if __name__=='__main__':
    rename()
    cal_all_mAP(alpha)
    draw_mAP()
    
