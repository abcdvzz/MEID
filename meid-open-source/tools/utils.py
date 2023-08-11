import numpy as np

def find_class_by_name(name, modules):
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def get_indices(_list, head=500, tail=100, dataset='VideoLT'):

    f = open(_list, 'r')
    lines = [line.strip() for line in f.readlines()]
    _list = np.zeros([len(lines)])
    head_indices = []
    tail_indices = []
    medium_indices = []
    for i in range(len(lines)):
        nums = float(lines[i].split(' ')[2])
        if nums <= tail:
            tail_indices.append(i)
        elif nums > tail and nums <= head:
            medium_indices.append(i)
        else:
            head_indices.append(i)
    return [tail_indices, medium_indices, head_indices]

def get_indices_test(_list, head=500, tail=100):
    f = open(_list, 'r')
    lines = [line.strip() for line in f.readlines()]
    _list = np.zeros([len(lines)])
    head_indices = []
    tail_indices = []
    medium_indices = []
    for i in range(len(lines)):
        nums = float(lines[i].split(' ')[2])
        if nums <= tail:
            tail_indices.append(i)
        elif nums > tail and nums <= head:
            medium_indices.append(i)
        else:
            head_indices.append(i)
    return [tail_indices, medium_indices, head_indices]

def get_exact_order(_list, head=500, tail=100):
    f = open(_list, 'r')
    lines = [line.strip() for line in f.readlines()]
    _list = np.zeros([len(lines)])
    tail_dic = {}
    medium_dic = {}
    head_dic = {}
    for i in range(len(lines)):
        nums = float(lines[i].split(' ')[2])
        if nums <= tail:
            tail_dic[i]=nums
        elif nums > tail and nums <= head:
            medium_dic[i]=nums
        else:
            head_dic[i]=nums
    new_tail = sorted(tail_dic.items(),key = lambda d:d[1], reverse=True)
    new_medium = sorted(medium_dic.items(),key = lambda d:d[1], reverse=True)
    new_head = sorted(head_dic.items(),key = lambda d:d[1], reverse=True)
    # return [tail_dic, medium_dic, head_dic]
    return [new_tail,new_medium,new_head]

if __name__=='__main__':
    aaa = get_exact_order('/home/yzbj10/Work/lxj/videolt/labels/count-labels-train.lst')
