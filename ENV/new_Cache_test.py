import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import random
import json
import csv, sys, ast
from absl import app
from absl import logging
random.seed(1234)

class Cache(object):
    def __init__(self, file_name, req_size_file_name, cache_size, need_feature, new_req_feature, skip_hit, skip_evict, reward_method, penalty_parameter):
        self.file_name = file_name
        self.need_feature = need_feature
        self.new_req_feature = new_req_feature
        
        self.skip_hit = skip_hit
        self.skip_evict = skip_evict
        self.reward_method = reward_method
        self.penalty_parameter = penalty_parameter

        # build cache
        self.cache_size = cache_size
        self.init_cache()

        ## request
        # load requests
        self.requests = self.load_data(self.file_name)
        self.requests_len = len(self.requests)

        # load req size
        self.req_to_size = self.load_req_size(req_size_file_name)
        
        ## init request
        self.init_request()

        ## address kind len
        self.onehot_len = max(self.requests)

        self.hit_history = []
        self.skip_evict_time = 0
        self.episode_count = 0
        
        ## gds para
        self.H = {}
        self.L = 0
        self.C = 1

    ## load requests
    def load_data(self, file_name):
        tmp = file_name.split('.')
        if tmp[1] == "csv":
            test_data = []
            with open(os.path.join(sys.path[0], "dataset/req_trace/", file_name), "r") as f:
                rows = csv.reader(f)
                for row in rows:
                    test_data.append(int(row[0])) 
        else:
            f = open("dataset/req_trace/"+file_name, "r")
            test_data = eval(f.read())
            f.close()
        logging.info("load data path: %s",f.name)
        return test_data

    def load_req_size(self, file_name):
        with open(("dataset/req_data/"+file_name+".txt"), "r") as f:
            a = f.read()
            req_to_size = ast.literal_eval(a)
        return req_to_size

    def save_hit_history(self, policy, path):
        ## save hit history
        # create folder
        try: 
            os.mkdir(path)
            print("create folder", path)
        except OSError: 
            print("folder exist", path)
            pass

        if policy == "DQN":
            # now = datetime.now()
            # current_time = now.strftime("%Y%m%d%H%M%S")
            f = open(path+"/"+str(self.episode_count)+".txt", "w")
            # print("save file:", path+"/"+str(self.episode_count)+".txt")
        else:
            f = open(path+"/"+policy+".txt", "w")
            # print("save file:", path+"/"+policy+".txt")
        # print(len(self.hit_history), self.get_hit_rate())
        f.write(str(self.hit_history))
        f.close()
    
    def pretty_print_hitrate(self):
        print('step {:5}, hit rate {:.5f}'.format(len(self.hit_history), self.hit_history.count(1) / len(self.hit_history)), end='\r')

    def show_cache(self):
        print("pointer", self.data_pointer)
        print("blocks", self.cache_blocks)
        print("freq", self.cache_freq)
        print("reqT", self.cache_req_time)
        print("size", self.cache_size)

    def stxt(self, file_name, *access_list):
        f = open('./look/'+file_name+'.txt', "a")
        f.write(str(access_list) +'\n')
        f.close()

    def init_cache(self):
        # init cache
        self.cache_blocks = []			## save tz/req
        self.cache_freq = []
        self.cache_rect = []
        self.cache_req_time = []		## Pr
        self.cache_req_size = []
        self.cache_miss_time = []		## Pm

    def init_request(self):
        # init request
        # new request
        self.new_request_card = None
        self.new_request_tzs = None
        # self.new_request_tg = None
        
        self.new_request = None
        # pointer
        self.data_pointer = -1
        # pre miss index (for wang reward use)
        self.pre_data_pointer = -1

    def reset(self):
        # episode += 1
        self.episode_count += 1
        # clear
        self.init_cache()
        self.init_request()
        self.hit_history.clear()
        self.skip_evict_time = 0

        self.get_next_miss()
        return self.get_observe()

    def req_to_onehot(self, req):
        onehot = [0] * self.onehot_len
        onehot[int(req)-1] = 1
        return onehot
    
    ## GDS use
    def gds_in(self):
        if (self.new_request not in self.H):
            self.H[self.new_request] = self.L + self.C / self.req_to_size[self.new_request]
    def gds_sc(self):
        # logging.info(f'after in: {self.data_pointer}, {self.H}, {self.cache_blocks}')
        a = []
        tmp_cache_block = self.cache_blocks.copy()
        
        while (self.cache_size - len(tmp_cache_block)) <= self.req_to_size[self.new_request]:
            
            min_con = max(self.H, key = self.H.get)
            for i in set(tmp_cache_block):
                if self.H[i] <= self.H[min_con]:
                    min_con = i
                    
            self.L = self.H[min_con]
            for _ in range(self.req_to_size[min_con]):
                tmp_cache_block.pop(tmp_cache_block.index(min_con))
            a.append(min_con)
        self.H[self.new_request] = self.L + self.C / self.req_to_size[self.new_request]
        return a
        
        
        
    def step(self, action):
        if self.skip_evict and action == 0: # not evict 可不驅逐
            self.skip_evict_time += 1
            reward = self.get_reward_noEvict() ##if self.reward_method != None else 0
            done, info = self.get_next_miss()	# get next miss request. if the request ends, done is True
            done = not done
        else: 								# evict
            if self.skip_evict: action -= 1
            ## method == our 時 這裡為0
            reward = self.get_reward(action, now_victim=self.cache_blocks[action], method=self.reward_method) if self.reward_method != None else 0
            if not self.remove_cache(action): 	# invalid action
                raise "invalid action"
            else:
                if self.has_enough_space(): # has enough space
                    self.insert_cache()				# insert new cache (# store the content at edge)
                    self.pre_data_pointer = self.data_pointer # record the previous miss time
                    ## 沒放滿給 penalty
                    # pen = self.penalty_parameter * (self.cache_size - len(self.cache_blocks))
                    if self.reward_method == None and len(self.cache_blocks) < self.cache_size: reward -= self.penalty_parameter
                    done, info = self.get_next_miss()	# get next miss request. if the request ends, done is True
                    done = not done
                    ## upset cache content
                    # self.upset_cache()
                else: 						# not enough free space 
                    # print("not enough")
                    # self.show_cache()
                    done = False
                    info = 'not enough'
        ## info{hit, miss, not enough}
        return self.get_observe(), reward, done, info
    
    def set_next_request(self):
        # data_pointer will stop in current request
        self.data_pointer += 1
        # logging.info(f'{self.data_pointer == self.requests_len}')
        # check out of range
        if self.data_pointer == self.requests_len:
            return False
        self.new_request = self.requests[self.data_pointer]
        return True

    def has_enough_space(self):
        # check free space
        return (self.cache_size - len(self.cache_blocks)) >= self.req_to_size[self.new_request]

    def insert_cache(self):
        for c in range(self.req_to_size[self.new_request]):
            self.cache_blocks.append(self.new_request)
            self.cache_freq.append(1)
            self.cache_rect.append(0)
            self.cache_req_time.append(self.data_pointer)
            self.cache_req_size.append(self.req_to_size[self.new_request])
            self.cache_miss_time.append(self.data_pointer)

    def pop_cache(self, index):
        self.cache_blocks.pop(index)
        self.cache_freq.pop(index)
        self.cache_rect.pop(index)
        self.cache_req_time.pop(index)
        self.cache_req_size.pop(index)
        self.cache_miss_time.pop(index)			## Pm
    
    def update_cache(self):
        index = self.cache_blocks.index(self.new_request)
        self.cache_freq[index] += 1
        self.cache_req_time[index] = self.data_pointer
        if self.req_to_size[self.new_request] == 2:
            self.cache_freq[index+1] += 1
        if self.req_to_size[self.new_request] == 3:
            self.cache_freq[index+2] += 1
        
        # req_size = self.req_to_size[self.new_request]
        # for i in range(1, req_size):
        #     self.cache_freq[index+i] += 1
        #     self.cache_req_time[index+i] = self.data_pointer
        
        # self.cache_req_time[index] = self.data_pointer
        if self.req_to_size[self.new_request] == 2:
            self.cache_req_time[index+1] = self.data_pointer
        if self.req_to_size[self.new_request] == 3:
            self.cache_req_time[index+2] = self.data_pointer


    def remove_cache(self, index): ## action 
        if index < len(self.cache_blocks):
            
            victim_req = self.cache_blocks[index]
            try:
                for _ in range(self.req_to_size[victim_req]):
                    index = self.cache_blocks.index(victim_req)
                    self.pop_cache(index)
                return True
            except:
                raise "No Cache Index!"
        else: return False # invalid action
            
    def get_reward(self, action, now_victim, method='hit_change'): # for evict
        if action > len(self.cache_blocks): raise "invalid action"
        reward = None
        if method == 'pre_len_rank': # calculate the previous request rank
            reward = (sorted(self.cache_req_time).index(self.cache_req_time[action])+1) / self.cache_size
        elif method == 'pre_len': # calculate the previous request len
            reward = self.data_pointer - self.cache_req_time[action]
        elif method == 'hit_change':
            ## count(1) 計算 1 有幾個
            reward = (self.hit_history[-100:].count(1) - self.hit_history[-200:-100].count(1)) / 100
        elif method == 'hit_rate':
            reward = self.hit_history[-100:].count(1) / 100
        elif method == 'freq^-1':
            reward = 1 / self.cache_freq[action]
        elif method == '1-past_freq_l':
            pass
            # reward = 1 - self.get_req_pre_freq(self.find_tg(self.cache_blocks[action]), self.cache_size * 3)
        elif method == 'wang':
            now_miss_index = self.data_pointer
            # from "peihaowang" github
            ## start
            alpha = 0.5
            psi = 10
            mu = 1
            # Compute reward: R = hit reward + miss penalty
            reward = 0.0
            # find next miss_resource
            next_miss_index, next_miss_resource, now_miss_use_count = self.find_next_miss(now_miss_index)
            hit_count = next_miss_index - now_miss_index - 1
            # print(next_miss_index, next_miss_resource_tzs, now_miss_use_count, hit_count)
            reward += hit_count
            # if (no evict), not implement
            ## cal. swap-in reward
            reward += (alpha * now_miss_use_count)
            ## cal. swap-out penalty
            if now_victim == next_miss_resource:
                reward -= (psi / (hit_count + mu))
            # else (evict), not implement
            ## implement in self.get_reward_noEvict function
            ## end
        elif method == 'zhong':
            now_miss_index = self.data_pointer
            # now_miss_resource_card = self.requests[now_miss_index]
            now_cache_ = self.cache_blocks.copy()
            # from "peihaowang" github, implement zhong's reward function
            ## start
            short_reward = 1.0
            long_span = 100
            beta = 0.5
            # Compute reward: R = short term + long term
            reward = 0.0
            # Total count of hit since last decision epoch
            # find next miss_resource
            next_miss_index, next_miss_resource_tzs, now_miss_use_count = self.find_next_miss(now_miss_index)
            hit_count = next_miss_index - now_miss_index - 1
            if hit_count != 0: reward += short_reward
            # Long term
            start = now_miss_index
            end = now_miss_index + long_span
            if end > len(self.requests): end = len(self.requests)
            long_term_hit = 0
            next_reqs = self.requests[start : end]
            for req in next_reqs:
                if req in now_cache_:
                    long_term_hit += 1
            # reward += beta * long_term_hit / (end - start)
            reward += beta * long_term_hit
            ## end
        elif method == 'size':
            reward = self.cache_req_size[action] * 0.3
        # elif method == 'fitness':
        #     reward = 0.5 if len(self.new_request_tzs) == self.cache_Ccs[action] else 0
        elif reward == None: 
            # raise "reward method error"
            reward = 0
        return reward

    def get_reward_noEvict(self):			# for 'peihaowang' method use
        now_miss_index = self.data_pointer
        beta = 0.3
        reward = 0.0
        # find next miss_resource
        next_miss_index, _, _ = self.find_next_miss(now_miss_index)
        # self.save_txt("find_next", self.episode_count, ":", self.data_pointer, self.requests[self.data_pointer], now_miss_index, next_miss_index)
        hit_count = next_miss_index - now_miss_index - 1
        reward += hit_count
        reward += reward * beta
        return reward

    # def find_next_miss(self, now_miss_index): # for 'peihaowang' use, find next miss request info
    #     temp_cache_tz = self.cache_blocks.copy()
    #     now_miss_content_tzs = self.new_request_tzs
    #     use_count = 0
    #     temp_request_tzs = []
    #     index = now_miss_index + 1
    #     while index < self.requests_len:
    #         temp_request_card = self.requests[index]
    #         temp_request_tg = self.card_to_tg(temp_request_card)
    #         temp_request_tzs = self.tg_to_tzs(temp_request_tg)
    #         if temp_request_tzs[0] not in temp_cache_tz:
    #             # occur "cache miss"
    #             break
    #         elif temp_request_tzs[0] in now_miss_content_tzs:
    #             use_count += 1
    #         index += 1
    #     return index, temp_request_tzs, use_count

    def find_next_miss(self, now_miss_index): # for 'peihaowang' use, find next miss request info
        temp_cache_ = self.cache_blocks.copy()
        now_miss_content_ = self.new_request
        use_count = 0
        temp_request_ = 0
        index = now_miss_index + 1
        while index < self.requests_len:
            temp_request_ = self.requests[index]
            if temp_request_ not in temp_cache_:
                # occur "cache miss"
                break
            elif temp_request_ == now_miss_content_:
                use_count += 1
            index += 1
        return index, temp_request_, use_count

    def get_next_miss(self):
        while self.set_next_request():
            if self.data_pointer != self.requests_len:
                self.gds_in()
            if self.new_request not in self.cache_blocks: ## miss
                self.hit_history.append(0)
                if self.has_enough_space(): self.insert_cache()
                else: return True, 'miss'
            else:
                self.update_cache()
                self.hit_history.append(1)
                if not self.skip_hit: return True, 'hit'

        return False, 'end'

    def get_next_request_time(self):
        next_request_time = []
        for req in self.cache_blocks:
            for pointer in range(self.data_pointer, self.requests_len):
                if req == self.requests[pointer]:                                   # found
                    next_request_time.append(pointer)
                    break
                elif pointer + 1 == self.requests_len:								# do not request in the future
                    next_request_time.append(pointer)

        return next_request_time

    def get_pre_freq(self, tg):
        count = 0
        for pointer in range(self.data_pointer-1, min(0, self.data_pointer - 100), -1):
            if (self.requests[pointer]) == tg: count += 1
        return count

    def get_req_pre_freq(self, req, LEN):
        if self.data_pointer < LEN: return 0
        count = 0
        for pointer in range(self.data_pointer-1, self.data_pointer - LEN, -1):
            if self.requests[pointer] == req : count +=1
        return count / LEN

    # def get_pre_req_his(self, l):
    #     if self.data_pointer - l >= 0 and self.data_pointer < self.requests_len:
    #         arr = []
    #         for i in range(l):
    #             # print(self.data_pointer - i)
    #             card = self.requests[self.data_pointer - i]
    #             tg = self.card_to_tg(card)
    #             arr.append(list(self.tg_to_onehot(tg.split('tg')[1])))
    #         # print(arr)
    #         return np.array(arr)
    #     else:
    #         return np.array([[0] * self.onehot_len] * l)

    def norm_list(self, list):
        if max(list)-min(list) == 0: return [0] * len(list)
        else: return [(float(i)-min(list))/(max(list)-min(list)) for i in list]
        
    def get_feature(self):
        feature = []
        
        if self.need_feature['freq'] == True: 
            feature += self.cache_freq.copy()
            for _ in range(self.cache_size - len(self.cache_blocks)):
                feature.append(0)
        
        ## 上一次該內容的請求未命中 time epoch 到目前的 time epoch
        ## 1*6
        if 'Ffqr_norm' in self.need_feature and self.need_feature['Ffqr_norm'] == True:
            ## normalize the rank of Ffq
            tmp_freq = self.cache_freq.copy()
            for _ in range(self.cache_size - len(self.cache_blocks)):
                tmp_freq.append(0)
            sorted_tmp_freq = sorted(tmp_freq)
            rank = [sorted_tmp_freq.index(x) for x in tmp_freq]
            norm_rank = self.norm_list(rank)
            feature += norm_rank
        
        ## no norm
        if 'Ffq_norm' in self.need_feature and self.need_feature['Ffq_norm'] == True:
            ## the number of request time / (t - Pr + 1)
            tmp_freq = self.cache_freq.copy()
            # print("m", self.cache_miss_time)
            # print("b", tmp_freq)
            for i in range(len(tmp_freq)):
                tmp_freq[i] /= (self.data_pointer - self.cache_miss_time[i] + 1)
            for _ in range(self.cache_size - len(self.cache_blocks)):
                tmp_freq.append(0)
            # print("a", tmp_freq)
            feature += tmp_freq
        
        ## 不同 term 的 frequency
        ## 1*6
        if self.need_feature['past_freq_l'] == True: 	
            # find the past frequency of "req" (past len=cache_size x3)
            feature += [self.get_req_pre_freq(req, self.cache_size * 3) for req in self.cache_blocks.copy()]
            for _ in range(self.cache_size - len(self.cache_blocks)):
                feature.append(0)
        if self.need_feature['past_freq_m'] == True: 	
            # find the past frequency of "req" (past len=cache_size x2)
            feature += [self.get_req_pre_freq(req, self.cache_size * 2) for req in self.cache_blocks.copy()]
            for _ in range(self.cache_size - len(self.cache_blocks)):
                feature.append(0)
        if self.need_feature['past_freq_s'] == True: 	
            # find the past frequency of "req" (past len=cache_size x1)
            feature += [self.get_req_pre_freq(req, self.cache_size * 1) for req in self.cache_blocks.copy()]
            for _ in range(self.cache_size - len(self.cache_blocks)):
                feature.append(0)
                
        ## 
        if self.need_feature['Ccs'] == True: 
            feature += self.cache_req_size.copy()
            for _ in range(self.cache_size - len(self.cache_blocks)):
                feature.append(0)
        
        ## default = False
        if 'Fts_norm' in self.need_feature and self.need_feature['Fts_norm'] == True:
            ## normalize the rank of Fts
            tmp_ts = self.cache_req_size.copy()
            for _ in range(self.cache_size - len(self.cache_blocks)):
                tmp_ts.append(0)
            norm_tmp_ts = self.norm_list(tmp_ts)
            feature += norm_tmp_ts
        
        ## rect Frt
        if self.need_feature['rect'] == True: 
            feature += [self.data_pointer - reqT for reqT in self.cache_req_time]
            for _ in range(self.cache_size - len(self.cache_blocks)):
                feature.append(0)
        ## Frtr_norm(false)
        if 'Frtr_norm' in self.need_feature and self.need_feature['Frtr_norm'] == True:
            ## normalize the rank of Frt
            tmp_rect = [self.data_pointer - reqT for reqT in self.cache_req_time]
            for _ in range(self.cache_size - len(self.cache_blocks)):
                tmp_rect.append(0)
            sorted_tmp_rect = sorted(tmp_rect)
            rank = [sorted_tmp_rect.index(x) for x in tmp_rect]
            norm_rank = self.norm_list(rank)
            feature += norm_rank
        
        ## (Far) all request data in cache
        if self.need_feature['Far'] == True: 
            for req in self.cache_blocks.copy():
                feature += self.req_to_onehot(req)
            for _ in range(self.cache_size - len(self.cache_blocks)):
                for _ in range(self.onehot_len):
                    feature.append(0)

        ## cache remain capacity
        if self.need_feature['remains'] == True:
            tmp_rsize = []
            for _ in range(len(self.cache_blocks)):
                tmp_rsize.append(0)
            for _ in range(self.cache_size - len(self.cache_blocks)):
                tmp_rsize.append(1)
            feature += tmp_rsize

        return np.array(feature)

    def get_feature_req(self):
        feature = []
        if self.new_req_feature['Ccs'] == True:
            feature.append(self.req_to_size[self.new_request])
        ## Fcr: current request data
        if self.new_req_feature['Fcr'] == True:
            feature += self.req_to_onehot(self.new_request)
        if self.new_req_feature['freq'] == True: ## default = False
            feature.append(self.get_pre_freq(self.new_req_feature['tg']))
        return np.array(feature)

    def get_observe(self):
        return dict(
            c_step=self.data_pointer,
            item=self.cache_blocks.copy(),
            freq=self.cache_freq.copy(),
            ## rect 算現在time跟前面每一個 request time 差距
            rect=[self.data_pointer-r for r in self.cache_req_time.copy()],
            req_time=self.cache_req_time.copy(),
            size=self.cache_req_size.copy(),
            next_req=self.get_next_request_time(),  # disable it, if your hit rate is very high and dataset is very large
            feature=np.concatenate([self.get_feature(), self.get_feature_req()]),
        )


def load_parameter(filepath):
    with open(file=filepath, mode='r') as FILE:
        return json.load(FILE)

def main(_):
    ## load feature parameter
    feature_parameter = load_parameter('./dataset/parameter/features/Ffqr_norm/full_new_norm.json')
    logging.info("Feature parameter :\n %s", feature_parameter)
    ## setting DQN parameter
    need_feature = feature_parameter["need_feature"]
    new_req_feature = feature_parameter["new_req_feature"]
    env = Cache(
        file_name='1.txt',
        req_size_file_name='gen1',
        cache_size=5, 
        # feature
        need_feature=need_feature,
        # now request feature
        new_req_feature=new_req_feature,
        skip_hit=True,
        skip_evict=False,
        reward_method='our',
        penalty_parameter=0.15
    )
    env.reset()
if __name__ == '__main__':
    app.run(main)
    