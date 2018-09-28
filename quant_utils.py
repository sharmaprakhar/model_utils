import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2


def quantize_weights(wts, numClusters, viz_codeX):
    original_data=np.copy(wts)
    nz_idx=np.nonzero(original_data)
    nz_data=original_data[nz_idx]
    F=nz_data.flatten()
    F=F.reshape(-1,1)
    InitC=np.linspace(F.min(),F.max(),num=numClusters)
    codebook, codeX=kmeans2(F, InitC.reshape(-1,1), minit='matrix')
    
    if viz_codeX==1:
        print(codebook)
        print(len(codeX))
        # edges_hist=[x for x in range(numClusters+1)]
        # frq, edges = np.histogram(codeX,edges_hist)
        # print(frq,edges)
        # fig, ax = plt.subplots()
        # ax.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
        # plt.title("cluster value histogram")
        # plt.show()
    return codebook, codeX

def create_codeVal(codeX):
    code_val=np.zeros(codeX.shape)
    for idx,val in enumerate(codeX):
        code_val[idx]=codebook[val]
    new_wts=code_val.reshape(wts.shape)
    return new_wts

def disrcetize_wts(weight):
    min_wt = weight.min() 
    max_wt = weight.max()
    #find number of integer bits to represent this range
    int_bits = int(np.ceil(np.log2(max(abs(min_wt),abs(max_wt))))) 
    frac_bits = 7-int_bits #remaining bits are fractional bits (1-bit for sign)
    #floating point weights are scaled and rounded to [-128,127], which are used in 
    #the fixed-point operations on the actual hardware (i.e., microcontroller)
    quant_weight = np.round(weight*(2**frac_bits))
    #To quantify the impact of quantized weights, scale them back to
    # original range to run inference using quantized weights
    weight = quant_weight/(2**frac_bits)
    return weight