import pandas as pd
import numpy as np
from scipy.stats import cramervonmises_2samp

chat_id = 440047378

def solution(x: np.array, y: np.array) -> bool:
    p_value = cramervonmises_2samp(x, y).pvalue
    return p_value < 0.01
