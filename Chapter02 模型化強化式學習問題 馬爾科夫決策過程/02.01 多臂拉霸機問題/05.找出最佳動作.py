
import numpy as np

def get_best_arm(record):
    arm_index = np.argmax(record[:,1])
    return arm_index