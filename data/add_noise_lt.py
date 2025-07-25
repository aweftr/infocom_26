# %%
import numpy as np
import pandas as pd

df = pd.read_csv("Huawei-East-1-lt.csv")
# %%
# df["lt"] = df["lt"].apply(lambda x: x + np.random.lognormal(mean=np.log(x * 0.02), sigma=1))
df["lt"] = df["lt"].apply(lambda x: x + np.random.normal(x * 0.1, 1))
# np.random.normal(x * 0.02, )

df.to_csv("Huawei-East-1-ltGaussianNoise.csv", index=False)
# %%