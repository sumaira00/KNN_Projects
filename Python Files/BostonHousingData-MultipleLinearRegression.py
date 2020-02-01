import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

from sklearn.datasets import load_boston
boston_data = load_boston()

df=pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
s3
