#@_author:alexhu
# time: 2019,10.24
import numpy as np
import numba as nb
from numba import jit
import pandas as pd
import os
import logging
import pickle
from tqdm  import tqdm
import time
import ipdb
class evaluation():
    # this clss is caculate the csi pod far and Standard skill score(标准技巧评分-百分制,由本人提出)(sss)
    # the  parameters is output data and target data the size is s*b*c*h*w  it use in deeplearning model's output and if you just use size  two dimension matrix h*w it aslo useful
    # otherwise if you use the deeplearning model you must set the outputfile name such as '3layersConvGRU' the detype is str
    # why do i write this clss? it just practice ,if you want to judge your deeplearning model skill score i suggest you choose hko-7 project evaluation it is more complete but not easy to set in your project
    def __init__(self,target,output,name = 'defult'):
        self.x = target
        self.y = output
        self.name = str(name)
        self.threshold = [10,20,30,40,50]
        self.all_caculate = {'csi10':[],'csi20':[],'csi30':[],'csi40':[],'csi50':[],
               'pod10':[],'pod20':[],'pod30':[],'pod40':[],'pod50':[],
               'far10':[],'far20':[],'far30':[],'far40':[],'far50':[]}
        self.score = []
    #@jit
    def caculatea(self,pandas = False):
        assert len(self.x) == len(self.y)
        #x = np.random.randint(0,2,(3,3))
        #y = np.random.randint(0,2,(3,3))
        """
        all_caculate = {'csi10':[],'csi20':[],'csi30':[],'csi40':[],'csi50':[],
               'pod10':[],'pod20':[],'pod30':[],'pod40':[],'pod50':[],
               'far10':[],'far20':[],'far30':[],'far40':[],'far50':[]}"""
    
        for i,key in enumerate(self.all_caculate):
            #if i >4:
                #break
            if i <5:
                threshold = self.threshold[i]
            else:
                threshold = self.threshold[i%5]
            #print(threshold)
            bpred = (self.y >= threshold)
            btruth = (self.x >= threshold)
            bpred_n = np.logical_not(bpred)
            btruth_n = np.logical_not(btruth)
            #print(xx0[0,0,0,0,5])
            #### attack it
            #ipdb.set_trace()

   
            tp = np.logical_and(bpred, btruth).sum().astype(np.float64)

            fn = np.logical_and(bpred_n, btruth).sum().astype(np.float64)

            fp = np.logical_and(bpred, btruth_n).sum().astype(np.float64)

            csi = tp / (tp + fn + fp+0.00000001)
            pod = tp / (tp + fn+0.00000001)
            #print(pod)
            far = fp /(tp + fp+0.00000001)
            #print(key)
            if i <= 4:
                self.all_caculate[key].append(csi)
            elif i<= 9:
                self.all_caculate[key].append(pod)
            else:
                self.all_caculate[key].append(far)
        
        if  pandas == True:
            self.savepd_data(self.all_caculate,self.name)
        return self.all_caculate
        
    
    def  sss(self):
        # this funcation is caculate the sss which the per weight for different threshold,the 0.5 for 10 dbz becasue it can reflect the basic precipation's location ,and 0.25 for 30dbz 0.25 for 50dbz if you want to
        # prefer lay stress on some threshold you can change the weight ,ok lets do it
        # attention it just use each batchsize rather than all ,you need change the funcation's postion
        caculate = self.all_caculate
        if caculate['csi10'][0]>-0.1 and caculate['csi30'][0]>-0.1  and caculate['csi50'][0]>-0.1  :
            sorce = (0.5*np.array(caculate['csi10']).mean()+0.25*np.array(caculate['csi30']).mean()+0.25*np.array(caculate['csi50']).mean())*100.0
            batch_sss = np.clip(sorce,0.0,100.0)
            #print("process 1")
        elif caculate['csi10'][0]>-0.1  and caculate['csi30'][0]>-0.1 :
            sorce = (0.75*np.array(caculate['csi10'][0]).mean()+0.25*np.array(caculate['csi30'][0]).mean())*100.0
            #print("process 2")
            batch_sss = np.clip(sorce,0.0,100.0)
        elif caculate['csi10'][0]>-0.1 :
            sorce = (np.array(caculate['csi10']).mean())*100.0
            #print(ok)
            #print("process 3")
            batch_sss = np.clip(sorce,0.0,100.0)
        else:
            batch_sss = np.nan

        return batch_sss

    #@nb.jit()
    def savepd_data(self,all_caculate_data,path = None):
        all_caculate_data = all_caculate_data
        dir_path = os.path.join(path,'fxx.pkl')
        print(dir_path)
        #if not os.path.exists(dir_path):
            #os.makedirs(dir_path)
        # convert the dict to dataframe
        temp = pd.DataFrame.from_dict(all_caculate_data, orient='index')
        f = open(dir_path, 'wb')
        logging.info("Saving pd_data to %s" %path)
        pickle.dump(temp, f)
        f.close()


######### 根据方大佬的代码，修改后并入集成评分程序，只需要将其全部转化为@staticmethod方法即可就可以愉快的使用了

    @staticmethod
    def prep_clf(obs,pre, threshold=0.1):
        '''
        func: 计算二分类结果-混淆矩阵的四个元素
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        
        returns:
            hits, misses, falsealarms, correctnegatives
            #aliases: TP, FN, FP, TN 
        '''
        #根据阈值分类为 0, 1
        obs = np.where(obs >= threshold, 1, 0)
        pre = np.where(pre >= threshold, 1, 0)

        # True positive (TP)
        hits = np.sum((obs == 1) & (pre == 1))

        # False negative (FN)
        misses = np.sum((obs == 1) & (pre == 0))

        # False positive (FP)
        falsealarms = np.sum((obs == 0) & (pre == 1))

        # True negative (TN)
        correctnegatives = np.sum((obs == 0) & (pre == 0))

        return hits, misses, falsealarms, correctnegatives

    @staticmethod
    def precision(obs, pre, threshold=0.1):
        '''
        func: 计算精确度precision: TP / (TP + FP)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        
        returns:
            dtype: float
        '''

        TP, FN, FP, TN = evaluation.prep_clf(obs=obs, pre = pre, threshold=threshold)

        return TP / (TP + FP)

    @staticmethod
    def recall(obs, pre, threshold=0.1):
        '''
        func: 计算召回率recall: TP / (TP + FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        
        returns:
            dtype: float
        '''

        TP, FN, FP, TN = evaluation.prep_clf(obs=obs, pre = pre, threshold=threshold)

        return TP / (TP + FN)


    @staticmethod
    def ACC(obs, pre, threshold=0.1):
        '''
        func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        
        returns:
            dtype: float
        '''

        TP, FN, FP, TN = evaluation.prep_clf(obs=obs, pre = pre, threshold=threshold)

        return (TP + TN) / (TP + TN + FP + FN)

    @staticmethod
    def FSC(obs, pre, threshold=0.1):
        '''
        func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
        '''
        precision_socre = precision(obs, pre, threshold=threshold)
        recall_score = recall(obs, pre, threshold=threshold)

        return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))
    
    @staticmethod
    def TS(obs, pre, threshold=0.1):
    
        '''
        func: 计算TS评分: TS = hits/(hits + falsealarms + misses) 
            alias: TP/(TP+FP+FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        returns:
            dtype: float
        '''

        hits, misses, falsealarms, correctnegatives = evaluation.prep_clf(obs=obs, pre = pre, threshold=threshold)

        return hits/(hits + falsealarms + misses) 
    

    @staticmethod
    def ETS(obs, pre, threshold=0.1):
        '''
        ETS - Equitable Threat Score
        details in the paper:
        Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of
        radar-derived precipitation with model-derived winds.
        Advances in Geosciences,10, 77–83. https://doi.org/10.5194/adgeo-10-77-2007
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                (rain/no rain)
        Returns:
            float: ETS value
        '''
        hits, misses, falsealarms, correctnegatives = evaluation.prep_clf(obs=obs, pre = pre,
                                                            threshold=threshold)
        num = (hits + falsealarms) * (hits + misses)
        den = hits + misses + falsealarms + correctnegatives
        Dr = num / den

        ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)

        return ETS

    @staticmethod
    def FAR(obs, pre, threshold=0.1):
        '''
        func: 计算误警率。falsealarms / (hits + falsealarms) 
        FAR - false alarm rate
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                (rain/no rain)
        Returns:
            float: FAR value
        '''
        hits, misses, falsealarms, correctnegatives = evaluation.prep_clf(obs=obs, pre = pre,
                                                            threshold=threshold)

        return falsealarms / (hits + falsealarms)

    @staticmethod
    def MAR(obs, pre, threshold=0.1):
        '''
        func : 计算漏报率 misses / (hits + misses)
        MAR - Missing Alarm Rate
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                (rain/no rain)
        Returns:
            float: MAR value
        '''
        hits, misses, falsealarms, correctnegatives = evaluation.prep_clf(obs=obs, pre = pre,
                                                            threshold=threshold)

        return misses / (hits + misses)


    @staticmethod
    def POD(obs, pre, threshold=0.1):
        '''
        func : 计算命中率 hits / (hits + misses)
        pod - Probability of Detection
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                (rain/no rain)
        Returns:
            float: PDO value
        '''
        hits, misses, falsealarms, correctnegatives = evaluation.prep_clf(obs=obs, pre = pre,
                                                            threshold=threshold)

        return hits / (hits + misses)
    
    @staticmethod
    def BIAS(obs, pre, threshold = 0.1):
        '''
        func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses) 
            alias: (TP + FP)/(TP + FN)
        inputs:
            obs: 观测值，即真实值；
            pre: 预测值；
            threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
        returns:
            dtype: float
        '''    
        hits, misses, falsealarms, correctnegatives = evaluation.prep_clf(obs=obs, pre = pre,
                                                            threshold=threshold)

        return (hits + falsealarms) / (hits + misses)

    @staticmethod
    def HSS(obs, pre, threshold=0.1):
        '''
        HSS - Heidke skill score
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): pre
            threshold (float)  : threshold for rainfall values binaryzation
                                (rain/no rain)
        Returns:
            float: HSS value
        '''
        hits, misses, falsealarms, correctnegatives = evaluation.prep_clf(obs=obs, pre = pre,
                                                            threshold=threshold)

        HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
        HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
                (misses + falsealarms)*(hits + correctnegatives))

        return HSS_num / HSS_den

    @staticmethod
    def BSS(obs, pre, threshold=0.1):
        '''
        BSS - Brier skill score
        Args:
            obs (numpy.ndarray): observations
            pre (numpy.ndarray): prediction
            threshold (float)  : threshold for rainfall values binaryzation
                                (rain/no rain)
        Returns:
            float: BSS value
        '''
        obs = np.where(obs >= threshold, 1, 0)
        pre = np.where(pre >= threshold, 1, 0)

        obs = obs.flatten()
        pre = pre.flatten()

        return np.sqrt(np.mean((obs - pre) ** 2))










if __name__ == '__main__':
    #%matplotlib inline
    import matplotlib.pyplot as plt

    # make a toy datasets for test class
    path = r'C:/Users/alexanderhu/Desktop/'
    fuck1 = {'csi10':[],'csi20':[],'csi30':[],'csi40':[],'csi50':[],
               'pod10':[],'pod20':[],'pod30':[],'pod40':[],'pod50':[],
               'far10':[],'far20':[],'far30':[],'far40':[],'far50':[]}
    x =[]
    fuck1 = pd.DataFrame.from_dict(fuck1)
    fuck1 = fuck1.stack().unstack(0)
    
    @jit
    def nbxx():
        for i in tqdm(range(10)):
            targetdata = np.random.randint(0,60,(10,2,1,480,480))
            outputdata = np.random.randint(0,60,(10,2,1,480,480))
        #gamma = np.random.randn(10,2,1,480,480)
        #targetdata =targetdata*gamma
            fuck ='C:/Users/Administrator/Desktop/'
            #outputdata = outputdata*gamma
            sexsixsix = evaluation(targetdata,outputdata,fuck)
            k = sexsixsix.caculatea()
            #print(k)
            x.append(sexsixsix.sss())
            k = pd.DataFrame.from_dict(k)
            #k = k.stack().unstack(0)
            print(k)
            fuck1 =fuck1.append(k)
            """
            for key in fuck1:
                temp = k[key][:]
                #print (temp)
                fuck1[key].append(temp[0])"""

                
    def savepd_data(all_caculate_data,path = None):
        all_caculate_data = all_caculate_data
        dir_path = os.path.join(path,'fxx.pkl')
        print(dir_path)
        #if not os.path.exists(dir_path):
            #os.makedirs(dir_path)
        # convert the dict to dataframe
        temp = pd.DataFrame.from_dict(all_caculate_data, orient='index')
        f = open(dir_path, 'wb')
        logging.info("Saving pd_data to %s" %path)
        pickle.dump(temp, f)
        f.close()
    begin = time.time()   
    nbxx()
    end = time.time()
    t = end -begin
    print(t)
    savepd_data(fuck1,fuck)
    #print(x)
    data = pd.read_pickle('C:/Users/Administrator/Desktop/fxx.pkl')
    data.stack()
    data.unstack(0)
    plt.figure()
    data.plot()
    plt.show()
    data.to_csv('C:/Users/Administrator/Desktop/fxx.csv')
