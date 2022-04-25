import random
import numpy as np
from collections import Counter
from alias import create_alias_tables, alias_sample


samples = ['a', 'b', 'c', 'd'] # 采样数组
probs = [0.3, 0.4, 0.2, 0.1] # 采样概率 
print("原始分布：")
print(probs)

s_num = 10000000 # 采样的数量
print("采样数量：", s_num)

print("【使用numpy方法实现非均匀采样】")
print("——————————————————————————————————————")
s = np.random.choice(samples, size=s_num, p=probs)
cnt_dict = Counter()
for item in s:
    cnt_dict[item] += 1
# 做归一化
cnt_list = cnt_dict.most_common(s_num)
prob_list = [(k, v / s_num) for k, v in cnt_list]
print(prob_list)


print("【使用前缀和+二分实现非均匀采样】")
print("——————————————————————————————————————")

# 计算前缀和数组
S = []
t = 0
for p in probs:
    t += p
    S.append(t)
print("前缀和数组", S)

# 使用二分法在前缀和中采样
def bs(random_num):
    l, r = 0, len(S) -1
    while l < r:
        mid = (l + r) // 2
        if S[mid] >= random_num: 
            r = mid
        else: 
            l = mid + 1
    return  r

cnt_dict = Counter()
for _ in range(s_num):
    a = random.uniform(0, 1)
    cnt_dict[samples[bs(a)]] += 1
    
cnt_list = cnt_dict.most_common(s_num)
prob_list = [(k, v / s_num) for k, v in cnt_list]
print(prob_list)


print("【使用Alias方法实现非均匀采样】")
print("——————————————————————————————————————")
accept_prob, alias_index = create_alias_tables(probs)

cnt_dict = Counter()
for _ in range(s_num):
    a = alias_sample(accept_prob, alias_index)
    cnt_dict[samples[a]] += 1
cnt_list = cnt_dict.most_common(s_num)
prob_list = [(k, v / s_num) for k, v in cnt_list]
print(prob_list)
