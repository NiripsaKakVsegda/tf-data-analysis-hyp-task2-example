import pandas as pd
import numpy as np
from hyppo.ksample import MMD

chat_id = 440047378

def solution(x: np.array, y: np.array) -> bool:
    p_value = MMD(compute_kernel="laplacian", gamma=10).test(x, y)[1]
    return p_value < 0.01
