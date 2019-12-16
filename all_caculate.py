import pandas as pd
import matplotlib.pyplot as plt
import evaluation as ev
import pickle
##加载npy 放到列表里
dataframe = {'csi10':[],'csi20':[],'csi30':[],'csi40':[],'csi50':[],
               'pod10':[],'pod20':[],'pod30':[],'pod40':[],'pod50':[],
               'far10':[],'far20':[],'far30':[],'far40':[],'far50':[]}
dataframe = pd.DataFrame.from_dict(dataframe)
data[i] = np.load()
for i in range():
    ev = ev(data[i][:,:480],data[i][:,480:])
    per_scorce =ev.caculatea()
    per_scorce= pd.DataFrame.from_dict(per_scorce)
    dataframe = dataframe.append(per_scorce)

pkl_path = ''
f = open(pkl_path, 'wb')
#print(dataframe)
pickle.dump(dataframe, f,protocol=pickle.HIGHEST_PROTOCOL)
f.close()



xx = pd.read_pickle(pkl_path)
##这个好像是滤去0.0值时间太久有点记不清了
xx = xx.where(xx>0.001)
#xx.loc['convgru']['csi10'].plot(kind='hist')
b= xx.loc['convlstm']['far10']
c = xx.loc['convgru']['csi10']
w = xx.loc['trijgru']['csi10']
z = xx.loc['conv3d']['csi10']
fk = pd.concat([b,c,w,z],axis=1)
fk.columns= ['convlstmcsi10','convgrucsi10','trijgrucsi10','conv3dcsi10']
#%matplotlib inline
fk.mean()
print(fk)
x = xx.loc['convgru']['csi10']
x.plot()
