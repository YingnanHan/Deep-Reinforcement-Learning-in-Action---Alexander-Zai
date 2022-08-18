
def update_record(record,action,r):
    r_ave = (record[action,0]*record[action,1]+r)/(record[action,0]+1) # 算出新的平均值
    record[action,0] += 1 # action號機台的拉桿次數加1
    record[action,1] = r_ave # 更新該機台的平均獎金
    return record