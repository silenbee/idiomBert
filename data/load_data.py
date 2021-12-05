import numpy as np
import json

# weibo_senti_100k
def load_senti_data(data_file, data_type):
    res = []
    with open(data_file, "r", encoding="utf-8") as rf:
        line = rf.readline()
        while line:
            t = line.split("\t")
            if len(t) != 2:
                print(t)
                line = rf.readline()
                continue
            res.append((t[0], t[1].replace("\n", "")))
            line = rf.readline()
    return np.array(res)

#chid_standard_data
def load_chid_data(data_file):
    dataList = []
    with open(data_file, "r", encoding="utf-8") as rf:
        line = rf.readline()
        while line:
            split_line = line.split("\t")
            content = split_line[0]
            if content == "" or content.find("#idiom#") == -1:
                line = rf.readline()
                continue
            content = content.replace("#idiom#", "[MASK]")
            mask_position = content.find("[MASK]")
            # provent for sentence not contain mask
            if mask_position >= 158:
                content = content[mask_position-158+5:]
            candidates = [int(i) for i in split_line[1].split(",")]
            label = int(split_line[2])
            dataList.append((content, candidates, label))
            line = rf.readline()
    return np.array(dataList)

