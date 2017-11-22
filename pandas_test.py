import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
s=pd.Series([1,2,3,4,5,6,'c'])
dates=pd.date_range('20171111',periods=5)


#df = pd.DataFrame(np.random.randn(5,4),index=dates,columns=list('ABCD'))

df = pd.DataFrame({'total_bill': [16.99, 10.34, 23.68, 23.68, 24.59],
                   'tip': [1.01, 1.66, 3.50, 3.31, 3.61],
                   'sex': ['Female', 'Male', 'Male', 'Male', 'Female']})
df2 = pd.DataFrame({'counts': [17.99, 11.34, 24.68, 23.68, 25.59],
                   'tops': [2.01, 2.66, 2.50, 2.31, 2.61]})

df11 = pd.DataFrame({'total_bill': [16.99, 10.34, 23.68, 23.68, 24.59],
                   'tip': [1.01, 1.66, 3.50, 3.31, 3.61],
                   'sex': ['Female', 'Male', 'Male', 'Male', 'Female'],
                   'name':['name1','name2','name3','name4','name5']})
df22 = pd.DataFrame({'counts': [17.99, 11.34, 24.68, 23.68, 25.59],
                   'tops': [2.01, 2.66, 2.50, 2.31, 2.61],
                   'namesss':['name1','name2','name3','name4','name5']})
#df3 =pd.DataFrame(json.loads(json_test))
#df2 = pd.DataFrame({"A":1.,
#                    "B":pd.Timestamp('20171111'),
#                    "C":pd.Series(1,index=list(range(4)),dtype="float32"),
#                    "D":np.array([3]*4,dtype="int32"),
#                    "E":"foo",
#                    "F":pd.Categorical(['test','train','test','train'])
#               })

print(df.head())
#print(df11.head())
#print(df22.head())
print("*****************************************************")

####自定义##################################################
#map(func)，为Series的函数，DataFrame不能直接调用，需取列后再调用；
#apply(func)，对DataFrame中的某一行/列进行func操作；
#applymap(func)，为element-wise函数，对每一个元素做func操作
#print(df['tip'].map(lambda x: x+1))
#print(df[['total_bill','tip']].apply(sum))
print(df.applymap(lambda x: x.upper() if type(x) is str else x))
####sql操作replace##########################################
# 全局
#print(df.replace(to_replace='Female',value='gilr'))
# dict方式
#print(df.replace({'sex':{'Female':'girl'}}))
#where 过滤
#df.loc[df.sex=='Male','sex']='boy'
#print(df.head())

####sql操作top##########################################
#全局
#print(df.nlargest(3,columns=['tip']))
#分组sex
#print(df.assign(rn=df.sort_values(['tip'],ascending=False).groupby('sex').cumcount()+1).query('rn<3').sort_values(['sex','rn']))
#print(df.assign(rn=df.sort_values(['tip'],ascending=False).groupby('sex').cumcount()))
#print(df.assign(rn=df.sort_values(['tip'],ascending=False).groupby('sex').cumcount()).query('rn<2').sort_values(['sex','rn']))
#print(df.assign(rn=df.groupby('sex')['tip'].rank(method='first',ascending=False)).query('rn<3').sort_values(['sex','rn'])[['sex','tip','total_bill']])


####sql操作order##########################################
# 排序 False 降序 Ture 升序
#print(df.sort_values(['tip','total_bill'],ascending=[True,False]))

####sql操作json##########################################
##支持left、right、inner、full outer 四种
#print((df.join(df2,how='le ft'))[df['sex']== 'Male']) #按默认的index进行
#print(pd.merge(df11,df22,how='left',left_on='name',right_on='namesss'))

####sql操作as############################################
# 别名操作
#df.columns=['sexes','top','nums'] #替换1
#df.rename(columns={'sex':'sexxx'},inplace=True)#替换2

####sql操作group##########################################
#group 配合合计函数使用(count,size 统计分类个数)
#print(df.groupby('sex').size())
#print(df.groupby('sex').count())
#print(df.groupby('sex')['tip'].count())
#agg 指定类型
#print(df.groupby('sex').agg({'tip':np.sum,'total_bill':np.max}))
#去重统计
#print(df.groupby('total_bill').agg({'sex':pd.Series.nunique,'tip':pd.Series.nunique}))

####sql操作distinct#######################################
## drop_duplicates
#print(df.drop_duplicates(subset=['total_bill'],keep='first',inplace=False))
'''
subset 选定的列做distinct 默认全部列
keep {'first','last',False} 保留重复元素的第一个 最后一个 全部不保留 
inplace False 返回新的df  True 将去重后的写入原df  默认为False
'''

####sql操作where#######################################
##df[df[colunm] boolean expr]
#print(df[df['sex']=='Male'])
#print(df[df['tip']>3])  #or print(df.query('tip>3'))

## and  &
#print(df[(df['sex']=='Male') & (df['tip']>3)])

## or  |
#print(df[(df['sex']=='Male') | (df['tip']>3)])

## in .isin
#print(df[df['sex'].isin(['Male','Female'])])

## not -
#print(df[-(df['sex'] == 'Male')])
#print(df[-(df['sex'].isin(['Male','Female']))])

## string function
#print(df[df['sex'].str.contains('^Fe')])

## where筛选后只有一行dataframe去其中一列的值
#print(df.loc[df['tip']>3,'sex'].values) #返回列表 
#print(df.loc[df['tip']>3,'sex':'total_bill'].values) #返回array数组  
#print(df.get_value(df.loc[df['tip']>3].index.values[0],'total_bill')) #get_value 获取其中一个的值

####sql操作select#######################################
#loc 基于列，选取特定行
#print(df.loc[1:3,['sex','total_bill']])  
#print(df.loc[1:3,'sex':'total_bill'])

#iloc 根据行列的索引
#print(df.iloc[1:3,1:3])#[1:3]$==[1:3)
#print(df.iloc[0:1,[0,2]])

#ix loc和iloc的合体 语法都支持
#print(df.ix[1:3,[1,2]])
#print(df.ix[1:3,0:2])

#at 根据index和colomn名获取指定元素
#print(df.at[3,'sex'])

#iat 根据index和colomn的index获取元素
#print(df.iat[3,0])

#简洁方式
#print(df[1:3])  #获取全部列的指定index行
#print(df[['sex','total_bill']]) #获取全部index的指定列

####DataFrame属性#######################################
#print(df.dtypes)  #df中各列的类型
#print(df.index)   #df索引
#print(df.columns) #列名的集合
#print(df.values)  #numpy.ndarray类型的数组
#print(df.shape)   #df的维度(indexs.num,columns.num)

##########################################################
#print(df.at[1,'tip'])
#print(df.iat[3,0])
#print(df.loc[:,['A','C']])
#print(df.loc[:,'A':'C'])
#print(np.nan)
#print(df.A)
#print(df[1:])
#print(df.describe())
#print(df.sort_index(axis=1,ascending=True))
#print(np.random.randn(4,3))