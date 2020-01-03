import pandas as pd
df = pd.DataFrame([[1,2],[3,4],[5,6]],columns=["A","asd"])
temp = pd.DataFrame([[1,3],[3,3],[5,3]],columns=["A","bcd"])
df = pd.concat([df, temp], axis=1)
print(df)