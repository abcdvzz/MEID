import os,sys
import numpy as np
file_dir = '/data/lxj/data/CharadesEgo/r101/'

import threading

lines = os.listdir(file_dir)

def fast_list2arr(data, offset=None, dtype=None):
    """
    Convert a list of numpy arrays with the same size to a large numpy array.
    This is way more efficient than directly using numpy.array()
    See
        https://github.com/obspy/obspy/wiki/Known-Python-Issues
    :param data: [numpy.array]
    :param offset: array to be subtracted from the each array.
    :param dtype: data type
    :return: numpy.array
    """
    num = len(data)
    out_data = np.empty((num,)+data[0].shape, dtype=dtype if dtype else data[0].dtype)
    for i in range(num):
        out_data[i] = data[i] - offset if offset else data[i]
    return out_data

def fun_(lines):
    for file in lines:
        aaa = np.load(file_dir+file)
        bbb = aaa[0]
        res = []
        for i in range(1,len(aaa)):
            bbb += aaa[i]
            ccc = aaa[i]-aaa[i-1]
            res.append(ccc)
        res.append(bbb)
        res = fast_list2arr(res)
        print(res.shape)
        new_name = file_dir+file.split('.')[0]+'_diff.npy'
        print(new_name)
        np.save(new_name, res)

threads, num_thread = [], 24
lines_count = 0
for i in range(num_thread):
    new_lines = lines[i::num_thread]
    lines_count += len(new_lines)
    ti = threading.Thread(target=fun_, args=(new_lines,))
    threads.append(ti)
for t in threads:
    t.start()
for t in threads:
    t.join()

print(len(lines))
print(lines_count)