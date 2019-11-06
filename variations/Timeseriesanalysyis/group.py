import pandas as pd
df = pd.read_csv("Power-Networks-LCL.csv")
gk = df.groupby('LCLid')
gk.first()
gk.get_group("MAC000002")
