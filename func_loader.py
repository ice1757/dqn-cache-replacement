import os
import csv
import json

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return True

def load_parameter(filepath):
    with open(file=filepath, mode='r') as FILE:
        return json.load(FILE)

def load_userdata(file_name):
    f = open("dataset/userdata/"+file_name+".txt", "r")
    datas =  f.readlines()
    f.close()
    tg = {}
    card = {}
    tg_card = {}
    for data in datas:
        split_data = data.split(';')
        tg[split_data[0]] = eval(split_data[1])
        tg_card[split_data[0]] = eval(split_data[2])
        for user in eval(split_data[2]):
            card[user] = split_data[0]
        # print(split_data[0], len(eval(split_data[1])), len(eval(split_data[2])))
    return tg, card, tg_card

def save_access_data(file_name, access_list):
    f = open('./dataset/'+file_name+'.txt', "w")
    f.write(str(access_list))
    f.close()
    print('successfully save dataset to', './dataset/'+file_name+'.txt')

def save_history(file_name, data):
    f = open('logs/'+file_name+'.txt', "w")
    f.write(str(data))
    f.close()

def load_history(filepath):
    f = open(filepath, "r")
    data = eval(f.readline())
    f.close()
    return data

def load_csv_history(filepaht):
    with open(filepaht, newline='') as csvfile:
        rows = csv.reader(csvfile)
        return [float(i) for i in list(rows)[0]]

def save_immediate_reward(path, data, method):
    if not check_folder(path): raise 'save_immediate_reward error 1'
    filename = path + "/" + str(method) +'_episode_immediate_reward'+'.csv'
    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()

def save_long_term_reward(path, data, method):
    if not check_folder(path): raise 'save_long_term_reward error 1'
    filename = path + "/" + str(method) +'_episode_long_term_reward'+'.csv'
    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()

def save_hit_rate(path, data, method):
    if not check_folder(path): raise 'save_hit_rate error 1'
    filename = path + "/" + str(method) + '.csv'
    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()

def save_config(path, csv_data):
    if not check_folder(path): raise 'save_config error 1'
    filename = path + "/" + 'config.csv'
    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for data in csv_data:
            writer.writerow(data)
    csvFile.close()


def cnt_diff_hit_rate(hit_history, change):
    hit_rate_r = []
    for i in range(1, change):
        hit_rate_r.append(hit_history[:int((len(hit_history)/change*i)-1)].count(1) / int((len(hit_history)/change*i)-1))
    return hit_rate_r

def savetxt(file_name, *value):
    f = open(file_name + '.txt', 'a')
    for i in value:
      f.write(str(i)+'\n-----------------\n')
    f.close()
    
def savetxt_look(file_name, *value):
    f = open('./look/' + file_name + '.txt', 'a')
    for i in value:
        f.write(str(i)+'\n-----------------\n')
    f.close()

def jwrt(file_name, data):
    with open(file_name+'.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)