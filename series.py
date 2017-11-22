import pandas as pd 
import numpy as np 
##汇总http://pandas.pydata.org/pandas-docs/stable/text.html
s = pd.Series(['A', 'B', 'Aaba', 'Baca', np.nan, 'CABA', 'dog'])
#print(s.str.len())
idx = pd.Index([' jack', 'jill ', ' jesse ', 'frank'])
#print(idx.str.upper())
df_str = pd.DataFrame(np.random.random((3, 2)), columns=[' Column A ', ' Column B '],index=range(3,6))
df_str.columns=  df_str.columns.str.strip().str.upper()
print(df_str)

print(df_str.columns)