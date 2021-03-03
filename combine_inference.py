import pickle 
import numpy as np 
import faiss 


# read the memory bank 
def mem_bank_read(path):
    f = open(path, 'rb')
    mem_bank = pickle.load(f)
    return mem_bank 


# construct the faiss index 
def faiss_index_construction(mem_bank): 
    key_list = np.array(mem_bank['key']).astype('float32')
    #print(key_list.shape)
    index = faiss.IndexFlatL2(key_list.shape[-1])
    index.add(key_list)
    return index 


# fast search for the top-k candidates
def fast_search(index, query, k=5): 
    return index.search(query, k)


if __name__ == "__main__": 
    mem_bank_path = 'data/mem_bank.pkl'
    mem_bank = mem_bank_read(mem_bank_path) 
    faiss_index_construction(mem_bank)