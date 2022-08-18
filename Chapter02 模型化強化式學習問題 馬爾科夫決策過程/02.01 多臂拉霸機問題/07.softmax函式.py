
import numpy as np

def softmax(av,tau=1.12): # av即動作的價值陣列，tau即溫度參數(預設值1.12)
    softm = (np.exp(av/tau))/np.sum(np.exp(av/tau))