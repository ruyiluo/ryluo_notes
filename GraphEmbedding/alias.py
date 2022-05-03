import random
from collections import Counter

def create_alias_tables(p_list):
    """
    :param p_list：概率列表，列表的和为1
    :return：返回一个accept_prob概率数组，和alias_index事件数组
    """
    N = len(p_list) # 事件总数
    # 初始化概率数组和事件数组
    accept_prob, alias_index = [0] * N, [0] * N
    
    # 定义两个队列
    small_queue, large_queue = [], []
    
    # 概率值乘以事件数，其结果与1进行比较，分别大于1和小于1的
    # 结果对应的索引添加到上述定义的两个队列中
    for i, p in enumerate(p_list):
        p *= N
        
        # 概率数组中的初试值，后面会重新分配
        accept_prob[i] = p
        
        if p < 1.0:
            small_queue.append(i)
        else:
            # 高度为1的柱子放在了large_queue中
            large_queue.append(i)
            
    # 遍历small_queue和large_queue, 每次都从中各取一个，用大的填补小的
    # 因为我们把高度为1的柱子放在了large中，所以，最终small_queue会为空
    while small_queue and large_queue:
        small_index = small_queue.pop()
        large_index = large_queue.pop()
        
        # 更新alias事件表, 因为我们是用大的填补小的，小的柱子才是
        # 对应着原来的事件，看填充过程的图片可能会更明显
        alias_index[small_index] = large_index
        
        # 更新高柱子的高度, 就是减去填补矮柱子到1的高度
        accept_prob[large_index] = accept_prob[large_index] - (1 - accept_prob[small_index])
        
        # 矮柱子的accept为什么不需要修改概率？是因为矮柱子留下的概率，就是当前位置对应事件的概率
        # 即使由高柱子变成了矮柱子，那也表示的是采样到当前柱子且采样到高柱子的概率是当前的概率值
        
        # 判断当前高柱子是否还是高柱子, 注意如果当
        if accept_prob[large_index] < 1.0:
            small_queue.append(large_index)
        else:
            large_queue.append(large_index)
    
    return accept_prob, alias_index
    

def alias_sample(accept_prob, alias_index):
    """
    :param accept_prob: 概率数组
    :param alias_index: 事件数组
    :return 返回采样的位置
    """
    # 先从所有柱子中选择一个柱子
    N = len(accept_prob)
    # 随机生成一个[0,(N-1)]的数
    random_n = random.randint(0, N-1)
    
    # 再随机生成一个 [0-1]之间的数
    random_i = random.random()
    
    # 判断当前柱子的accpet_prob值与生成的random_i的大小
    if random_i <= accept_prob[random_n]:
        return random_n

    return alias_index[random_n]