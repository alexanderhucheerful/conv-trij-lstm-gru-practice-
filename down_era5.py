from queue import Queue
from threading import Thread
import cdsapi
import time
from datetime import datetime,timedelta
import os,glob
from multiprocessing import Pool
download_dir = os.getcwd()
def download_ear5(allvar):
    print(allvar)
    variable = allvar[0]
    pressure_level= allvar[1]
    year= allvar[2]
    month= allvar[3]
    day= allvar[4]
    hour= allvar[5]
    file_path = os.path.join(download_dir,variable,pressure_level,year,month,day,variable+"_"+pressure_level+'hpa'+"_"+year+month+day+hour+".nc")
    if (not os.path.exists(file_path))  or (os.path.getsize(file_path) < 1024*1024*1.8) :
        c = cdsapi.Client()
        r = c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [variable],
                'pressure_level': [pressure_level],
                'year': year,
                'month': [month],
                'day': [day],
                'time': [hour+':00'],
                #'area' : [32, 118, 27, 124],
            },
            )
        r.download(os.path.join(download_dir,variable,pressure_level,year,month,day,variable+"_"+pressure_level+'hpa'+"_"+year+month+day+hour+".nc"))
        r.delete()
    return
    
#下载脚本 
class DownloadWorker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue
 
    def run(self):
        while True:
            # 从队列中获取任务并扩展tuple
            allvar = self.queue.get()
            download_ear5(allvar)
            self.queue.task_done()
#主程序 
def main():
    start_time  =  datetime(2017,1,1,0)  # 开始时间
    end_time = datetime(2020,1,1,0)      # 结束时间
    #创建列表，存储起止时间之间的所有时次
    datetime_list = []   
    while start_time <= end_time:
        datetime_list.append(start_time)
        start_time +=  timedelta(hours=1)
    #变量列表
    variable_list = ['geopotential', 'specific_humidity', 'temperature','u_component_of_wind', 'v_component_of_wind']
    #气压层列表
    pressure_level_list = ['100','200', '500','700', '850', '925', '1000']
    #创建allvar列表，内含variable,pressure_level,year,month,day,hour
    allvar = []
    for idatetime in datetime_list:  #非并行创建文件夹，同时填充allvar列表
        year = idatetime.strftime('%Y')
        month = idatetime.strftime('%m')
        day = idatetime.strftime('%d')
        hour = idatetime.strftime('%H')
        for pressure_level in pressure_level_list:
            for variable in variable_list:
                if not os.path.exists(os.path.join(download_dir,variable,pressure_level,year,month,day)):
                    os.makedirs(os.path.join(download_dir,variable,pressure_level,year,month,day))
                allvar.append([variable,pressure_level,year,month,day,hour])
    job_number= 12
    if len(allvar)%job_number==0:
        sub_mission_len = int(len(allvar)/job_number)
    else:
        sub_mission_len = int(len(allvar)/job_number)+1
    for i in range(sub_mission_len):
        if i*4+4  > len(allvar):
            allvar_part = allvar[i*job_number:]
        else:
            allvar_part = allvar[i*job_number:i*job_number+job_number]
        p = Pool(len(allvar_part))
        for i in range(len(allvar_part)):
            p.apply_async(download_ear5, args=(allvar_part[i],))
        #print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        #print('All subprocesses done.')
if __name__ == '__main__':
    main()