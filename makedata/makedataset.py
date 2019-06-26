#【データセットの作成】
# 時刻tの流入量を出力するための入力データセットを作成する

# ＜引数とそれに対応するデータセット作成のルール (論文と表記のルールが違うので注意（Rt,Dtなど）)＞
# __init__
#   Rt -> 時刻 t-1 から時刻 t-Rt の降水量（8か所）をデータセットに含める
#   rain_on_time -> 時刻 t の降水量（8か所）をデータセットに含める
#   Dt -> 時刻 t-1 から時刻　t-Dt の排水量（親松排水機場・鳥屋野潟排水機場）をデータセットに含める
#   drainage_on_time -> 時刻 t の排水量（親松排水機場・鳥屋野潟排水機場）をデータセットに含める
#   base_rain -> データセットに含める際の基準降水量

import sqlite3
from contextlib import closing
import pickle
import bz2
import pandas as pd
import datetime

class makedataset:
    def __init__(self,dbname,Rt,Dt,rain_on_time = True,drainage_on_time = True,base_rain = 12):
        self.dbname = dbname
        self.rt = Rt  #入力に含める降水量のタイムステップ
        self.dt = Dt  #入力に含める排水量のタイムステップ
        self.r_ont = rain_on_time  #出力対象時刻の降水量を含めるか
        self.d_ont = drainage_on_time  #出力対象時刻の排水量を含めるか
        self.br = base_rain #基準降水量

        self.dataset = []
        self.days = []
        self.reals = []
    def ptoz(obj):
        PROTOCOL = pickle.HIGHEST_PROTOCOL
        return bz2.compress(pickle.dumps(obj, PROTOCOL), 3)

    def MakeOneData(self,_id, c, data_table_name,rule,inflow_file):
        inflow = pd.read_csv(inflow_file,index_col=0)
        res = []
        fail = False
        #探索区間の設定　と　スイッチ配列の作成
        datalength = self.rt if self.rt > self.dt else self.dt
        if self.d_ont or self.r_ont:
            last_id = _id
            counter = datalength + 1

            # 降雨量と排水量を含めるためのスイッチ配列
            rain_switch = [True if i <= self.rt else False for i in range(datalength, 0,-1)]
            if self.r_ont:
                rain_switch.append(True)
            else:
                rain_switch.append(False)
            drainage_switch = [True if i <= self.dt else False for i in range(datalength, 0,-1)]
            if self.d_ont:
                drainage_switch.append(True)
            else:
                drainage_switch.append(False)
        else:
            last_id = _id -1
            counter = datalength
        _cond = "where _id >= " + (str)(_id - datalength) + " and _id <= " + (str)(last_id)
        c.execute("select * from " + data_table_name + " " + _cond)
        targetdata = c.fetchall()

        for (row,r_in,d_in) in zip(targetdata,rain_switch,drainage_switch):
            (i, d, o, h, ka, ku, m, n, r, s, t, do, dt, year, month, day,hour) = row

            #ル―ル検証
            if(rule(month)):
                time_t = datetime.datetime(year, month, day,hour = hour)
                if r_in:
                    res.append(o)
                    res.append(h)
                    res.append(ku)
                    res.append(m)
                    res.append(n)
                    res.append(r)
                    res.append(s)
                    res.append(t)

                if do is None:
                    do = 0
                if d_in:
                    dr = (do + dt) * 0.036
                    res.append(dr)

            else:
                #12月、1月、2月をデータに含めようとした時点でデータ作成失敗
                fail = True

        c.execute("select Date,Year,Month,Day,Hour from " + data_table_name + " where _id = " + str(_id))
        fuga = c.fetchall()[0]
        _date,t_year,t_month,t_day,t_hour = fuga
        time = datetime.datetime(t_year,t_month,t_day,t_hour)
        real = inflow.at[_date, "0"]
        return (res, time,real,fail)

    def __call__(self,rule,event_file_name,data_file_name,inflow_file):

        with closing(sqlite3.connect(self.dbname)) as conn:
            cur = conn.cursor()
            select_sql = "select * from " + event_file_name
            cur.execute(select_sql)
            for row in cur.fetchall():
                sd,ed,t,*hoge = row
                if t > self.br and ed != "":
                    _select_sd = "select _id  from " + data_file_name + " where Date = " + "\'" + sd + "\'"
                    _select_ed = "select _id  from " + data_file_name + " where Date = " + "\'" + ed + "\'"

                    cur.execute(_select_sd)
                    start_id,*hoge = cur.fetchall()[0]
                    cur.execute(_select_ed)
                    end_id, *hoge = cur.fetchall()[0]
                    for i in range(start_id,end_id):
                        res,day,real,fail = self.MakeOneData(_id = i,c= cur,data_table_name=data_file_name,rule = rule,inflow_file=inflow_file)
                        if not fail:
                            self.dataset.append(res)
                            self.days.append(day)
                            self.reals.append(real)

    def getdata(self,testyear):
        x_train = []
        x_test = []
        t_train = []
        t_test = []
        for dt,dy,rl in zip(self.dataset,self.days,self.reals):
            year = dy.year
            if year == testyear:
                x_test.append(dt)
                t_test.append(rl)
            else:
                x_train.append(dt)
                t_train.append(rl)
        return x_train,t_train,x_test,t_test