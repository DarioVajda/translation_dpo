from datasets import load_from_disk

def load_data():
    return load_from_disk('/ceph/hpc/data/s24o01-42-users/corpuses/wikipedia/wikipedia_en/train')

def selection(n, m, id):
    return [i for i in range(n) if i % (300) == id][:m]