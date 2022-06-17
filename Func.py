from collections import defaultdict
from absl import logging
import numpy as np
import random
from func_loader import load_userdata, savetxt, savetxt_look

class Temp_Memory(object):
    def __init__(self, memory_size, n_features, normalize_reward=True):
        self.memory_size = memory_size
        self.n_features = n_features
        self.temp_memory = np.zeros((memory_size, n_features * 2 + 2))
        ## defaultdict不會 raise key error
        self.temp_sa = defaultdict(dict)
        self.normalize_reward = normalize_reward

    def reset(self):
        del self.temp_sa
        self.temp_sa = defaultdict(dict)

    def get_key_if_len_gt(self, now_time, len_limit):
        transition_list = []
        ## get "temp_sa"'s key if len greater than "len_limit"
        for key, value in self.temp_sa.items():
            del_val_list = []
            for val in value:
                # print(self.temp_sa[key])
                time = self.temp_sa[key][val][1]
                if now_time - time > len_limit: # reuse distance greater then "len_limit"
                    index = self.temp_sa[key][val][0]
                    # retrieve from temp_memory
                    pre_transition = self.temp_memory[index]
                    pre_transition[-self.n_features-1] += np.log(len_limit) 		# assign reward
                    s, a, r, s_ = self.transition_decoder(pre_transition)
                    transition_list.append([s, a, r, s_])
                    # print(key, val, self.temp_sa[key][val], now_time - self.temp_sa[key][val][1])
            for val in del_val_list:
                del self.temp_sa[key][val]
        return transition_list

    def get_key(self, req):
        key_list = []
        for key, value in self.temp_sa.items():
            if req in value:
                key_list.append(key)
        return key_list if len(key_list) > 0 else None

    def store_tempSA(self, s, a, index, time):  ## s:new request、a:victim、index: mem size index、time:data point
        # savetxt('./look/st_sa', f'{s}, {a}, {index}, {time}, {self.temp_sa}')
        if s not in self.temp_sa:			## s never appeared / new req 沒出現
            self.temp_sa[s][a] = [index, time]
        elif a not in self.temp_sa[s]:		## s appeared, a never appeared / 
            self.temp_sa[s][a] = [index, time]
        else:
            self.temp_sa[s][a] = [index, time]

    def transition_decoder(self, transition):
        s = transition[:self.n_features]
        a = transition[self.n_features]
        r = transition[self.n_features+1]
        s_ = transition[-self.n_features:]
        return s, a, r, s_
    
    

    def store_temp_transition(self, s, a, r, s_, new_cache, victim, time, cache_block, cache_freq, his_req):
        if not hasattr(self, 'memory_counter_temp'):
            self.memory_counter_temp = 0
        ## remove the temp_sa with too long time step length
        # transition_list = self.get_key_if_len_gt(time, 100)
        # logging.info(f'time:{time}, {his_req}')
        def sum2():
            rw_sum = 0
            for i in set(cache_block):
                if i == new_cache:
                    continue
                try:
                    last_use = len(his_req) - his_req[::-1].index(i)-1
                except:
                    last_use = time
                rw_sum += time - last_use
            return rw_sum
        def sum3():
            rw_sum = 0
            try:
                for i in set(cache_block):
                    if i == new_cache:
                        continue
                    pres = self.get_key(i)
                    if pres:
                        for pre in pres:
                            lenp = time - self.temp_sa[pre][i][1]
                            # savetxt('sum3', f't:{time}, {evi_time}, {self.temp_sa[i]}')
                            rw_sum += lenp
            except:
                rw_sum = 10
            if rw_sum == 0:
                rw_sum = 10
            # savetxt_look('l_rw', f'{rw_sum}')
            return rw_sum
            
        transition_list = []
        if new_cache != victim:
            # miss
            ## store temp transition in temp_memory
            index = self.memory_counter_temp % self.memory_size
            transition = np.hstack((s, [a, r], s_))
            self.temp_memory[index, :] = transition
            
            self.memory_counter_temp += 1
            ## store the index of this transition
            self.store_tempSA(new_cache, victim, index, time)
            ## find previous transition (pre victim = now request)
            cn = 0
            pre_requests = self.get_key(new_cache)
            if pre_requests:
                for pre_request in pre_requests:
                    ptime = self.temp_sa[pre_request][new_cache][1]
                    pre_len = time - self.temp_sa[pre_request][new_cache][1]
                    
                    # print(pre_len)
                    if pre_len == 0:
                        print(time, pre_request, new_cache, victim)
                        print(self.temp_sa[pre_request][new_cache][1])
                        raise "zero"
                    pre_index = self.temp_sa[pre_request][new_cache][0]
                    del self.temp_sa[pre_request][new_cache]
                    pre_transition = self.temp_memory[pre_index]
                    if self.normalize_reward: 
                        # pre_len = pre_len if pre_len <= 100 else 100
                        # pre_len /= 100
                        # pre_len = (pre_len)
                        pre_len = np.log(pre_len)
                    
                    if cn < 1:
                        cf = sum2()
                        if cf == 0:
                            pre_transition[-self.n_features-1] -= 0
                        else:
                            pre_transition[-self.n_features-1] += np.log(cf)
                        cn+=1
                    
                    pre_transition[-self.n_features-1] += pre_len 		# assign reward
                    # pre_transition[-1] = new_cache.split('g')[1]
                    s, a, r, s_ = self.transition_decoder(pre_transition)
                    transition_list.append([s, a, r, s_])
                    # self.store_transition(s, a, r, s_)
                    # break
                    # raise "find"
        return transition_list

# class TestData(object):
#     def __init__(self, userdata_name, max_ug_size, max_tg_size):
#         self.test_data = []
#         self.timezone_group, self.card, self.tg_card = load_userdata(userdata_name)
#         self.tg_list = list(self.tg_card.keys())
#         self.card_list = list(self.card.keys())
#         self.max_ug_size = max_ug_size
#         self.max_tg_size = max_tg_size
#     def insert_pre(self, pre_len):
#         # 選前 pre_len 個出現過的卡號
#         if pre_len < len(self.test_data) and random.random() < 0.8:
#             self.test_data.append(self.test_data[random.randint(0, pre_len)])
#         else :
#             self.test_data.append(str(random.randint(1, len(self.card))))
#         return
#     def insert_large_ug(self):
#         # 選擇 user group 較大的 (bigger than max_ug_size // 2)
#         while True:
#             tg_index = random.choice(self.tg_list)
#             if len(self.tg_card[tg_index]) > self.max_ug_size // 2:
#                 self.test_data.append(str(random.choice(self.tg_card[tg_index])))
#                 break
#         return
#     def insert_tg_x(self, x):
#         # 選擇 timezone group = x 的
#         while True:
#             tg_index = random.choice(self.tg_list)
#             if len(self.timezone_group[tg_index]) == x:
#                 self.test_data.append(str(random.choice(self.tg_card[tg_index])))
#                 break
#         return
#     # def insert_large_tg(self):
#     #     # 選擇 time group 較大的 (2)
#     #     while True:
#     #         tg_index = chr(random.randint(97,97+len(self.timezone_group)-1))
#     #         if len(self.tg[tg_index]) < 3:
#     #             self.test_data.append(str(self.tg[tg_index][random.randint(0, len(self.tg[tg_index])-1)]))
#     #             break
#     #     return
#     def insert_LRU(self, lens):
#         rand = random.random()
#         if rand < 0.75 and len(self.test_data) > lens:
#             temp = self.test_data[-lens:]
#             temp.reverse()
#             for t in temp:
#                 if random.random() < 0.75:
#                     self.test_data.append(t)
#                 else:
#                     self.insert_tg_x(1)
#             # self.test_data += temp
#         else:
#             for i in range(lens):
#                 # self.insert_large_ug()
#                 if random.random() < 0.9:
#                     self.insert_tg_x(1)
#                 else:
#                     self.insert_tg_x(2)

#                 # tg_index = chr(random.randint(ord("a"),ord("a")+len(self.timezone_group)-1))
#                 # self.test_data.append(str(self.tg[tg_index][random.randint(0, len(self.tg[tg_index])-1)]))

#     def insert_random(self):
#         tg_index = random.choice(self.tg_list)
#         self.test_data.append(str(random.choice(self.tg_card[tg_index])))

#     def insert_probability(self, t_len=10000):
#         # next_prob_offset = 0.5
#         rank = self.card_list.copy()
#         rank_list = np.zeros((len(rank), len(rank)))
#         for i in range(len(rank_list)):
#             random.shuffle(rank)
#             pdf = 1 / np.power(rank, 1.3)
#             pdf /= np.sum(pdf)
#             # print(pdf[rank[0]-1])
#             rank_list[i] = pdf
#         # print(rank_list)
#         now = rank[0]
#         for i in range(t_len):
#             self.test_data.append(str(now))
#             if random.random() < 0.1:
#                 now = np.random.choice(self.card_list, size=1, p=rank_list[now-1])[0]
#             else: 
#                 now = rank[now-1]
    
#     def insert_zipf_tg(self, zipf_para=1.3, t_len=10000, change_rank=None):
#         # generate the zipf distribution data
#         print("chang_rank =", change_rank, "length =", t_len)
#         rank = [i+1 for i in range(len(self.tg_list.copy()))]
#         # print(rank)
#         # rank_list = np.zeros(len(rank))
#         # print(rank_list)
#         random.shuffle(rank)
#         # print(rank)
#         pdf = 1 / np.power(rank, zipf_para)
#         pdf /= np.sum(pdf)
#         # print(pdf)
#         data = []
#         if not change_rank: data = list(np.random.choice(self.tg_list, size=t_len, p=pdf))
#         else:
#             for i in range(t_len):
#                 if i % change_rank == 0:
#                     random.shuffle(rank)
#                     # print(rank)
#                     pdf = 1 / np.power(rank, zipf_para)
#                     pdf /= np.sum(pdf)
#                     # print(pdf)
#                 data.append(list(np.random.choice(self.tg_list, size=1, p=pdf))[0])
#         # conv tg to card
#         for i in range(len(data)):
#             data[i] = np.random.choice(self.tg_card[data[i]], size=1)[0]
#         self.test_data += data

#     def card_to_tg(self, card):
#         return self.card[int(card)]
    
#     def tg_to_tzs(self, tg):
#         return self.timezone_group[tg]
    
#     def tgx_ratio(self, tg_size):
#         # find the time zone group size == tg_size's ratio
#         count = 0
#         for user in self.test_data:
#             tg = self.card_to_tg(user)
#             tzs = self.tg_to_tzs(tg)
#             if len(tzs) == tg_size: count += 1
#             # print(len(tzs), tg, tzs, user)
#             # print(self.card[int(user)])
#         return count / len(self.test_data)