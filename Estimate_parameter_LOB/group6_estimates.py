
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # linear regression
from sklearn.linear_model import LinearRegression 

LOB5 = pd.read_csv("INTC_2012-06-21_34200000_57600000_orderbook_5.csv",
header=None,usecols= list(range(20)),names=["AskPrice1","AskSize1","BidPrice1","BidSize1",
                       "AskPrice2","AskSize2","BidPrice2","BidSize2",
                       "AskPrice3","AskSize3","BidPrice3","BidSize3",
                       "AskPrice4","AskSize4","BidPrice4","BidSize4",
                       "AskPrice5","AskSize5","BidPrice5","BidSize5"])
LOB5.loc[:,["AskPrice1","AskPrice2","AskPrice3","AskPrice4","AskPrice5",
"BidPrice1","BidPrice2","BidPrice3","BidPrice4","BidPrice5"]] = LOB5.loc[:,["AskPrice1","AskPrice2","AskPrice3","AskPrice4","AskPrice5",
"BidPrice1","BidPrice2","BidPrice3","BidPrice4","BidPrice5"]]/10000
messages = pd.read_csv("INTC_2012-06-21_34200000_57600000_message_5.csv",
header=None, names=["Time","Type","OrderID","Size","Price","Direction"])
# add timestamp to the Limit Order BOOK
LOB5["Time"] = messages.Time
messages.loc[:,"Price"] = messages["Price"]/10000
messages["midprice"] = (LOB5.AskPrice1 + LOB5.BidPrice1)/2

execution = messages.loc[(messages.Type == 4) | (messages.Type == 5),:]
execution.loc[:,"Market Order Type"] = execution.loc[:,"Direction"].apply(lambda x: 'Market Order - Buy' if x==-1 else 'Market Order - Sell')

# lob.png
ax = sns.barplot(x= LOB5.loc[10000,["BidPrice1","BidPrice2","BidPrice3","BidPrice4","BidPrice5","AskPrice1","AskPrice2","AskPrice3","AskPrice4","AskPrice5"]].to_list(), 
y=LOB5.loc[10000,["BidSize1","BidSize2","BidSize3","BidSize4","BidSize5","AskSize1","AskSize2","AskSize3","AskSize4","AskSize5"]].to_list(), palette=["red"]*5+["green"]*5)
ax.set_xlabel('Price', fontsize = 14)
ax.set_ylabel('Size', fontsize = 14)
plt.show()

sns.histplot(data=messages, x = "Type", hue="Direction",palette=sns.color_palette("tab10",2))
None

# midprice.png
sns.set(rc={'figure.figsize':(5,4)})
ax = sns.lineplot(data = execution.loc[execution.Time<=40000], x = "Time", y = "Price",ci=None,hue="Market Order Type",palette=sns.color_palette("tab10",2),alpha = 0.6) 
ax = sns.lineplot(data = messages.loc[messages.Time<=40000],x = "Time",y ="midprice",label="Midprice",ci=None,alpha = 0.3, color ="green")
ax.set_xlabel('Time (Seconds after Midnight)', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
# The excution of the Limit order orginates from the arrival of Market Order.
plt.legend() 
None

# Divide into 5-minute windows
interval_point = np.arange(execution.Time[0],execution.Time[581029]+300,step=300)
execution["interval"] = pd.cut(execution.loc[:,"Time"], bins= interval_point, right=True, labels=np.arange(5,5*78+1,step=5), retbins=False, precision=8, include_lowest=True, duplicates='raise', ordered=True)


MO_volume = execution.loc[:,["Market Order Type","interval","Size"]].groupby(["interval","Market Order Type"]).sum().reset_index(level="Market Order Type").pivot(columns="Market Order Type",values="Size")
last_midprice = execution.loc[:,["Time","midprice","interval"]].groupby("interval").tail(1).set_index("interval")
binned_data = last_midprice.join(MO_volume)

if "midprice_last" not in binned_data.columns:
    binned_data = binned_data.join(last_midprice.loc[:,"midprice"].shift(1),rsuffix="_last")
binned_data.dropna(inplace=True)
binned_data["price_change"] = binned_data.midprice - binned_data.midprice_last
binned_data["net order flow"] = binned_data["Market Order - Buy"] - binned_data["Market Order - Sell"]
x = binned_data["net order flow"]
y = binned_data["price_change"]
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"b = {slope}")

# regression_b.png
sns.set(rc={'figure.figsize':(5,4)})
ax = sns.regplot(data=binned_data,x="net order flow",y="price_change",ci=95)
ax.plot(x, intercept + slope*x, 'r', label='fitted line')
ax.set_xlabel('Net Order flow', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
plt.show()

# net_flow.png
ax= sns.displot(data=binned_data,x="net order flow",bins=10,kde=True,height=4,aspect=6/4)
ax.set_xlabels('Net Order_flow', fontsize = 14)
ax.set_ylabels('Count', fontsize = 14)
plt.show()

# residual.png
resid = binned_data.price_change - (intercept + slope*x)
ax = sns.displot(x=resid,bins=14,kde =True,height=4,aspect=6/4)
ax.set_xlabels("Residual",fontsize =16)
ax.set_ylabels("Count",fontsize = 14)
plt.show()

LOB5["rounded_time"] = LOB5.Time.apply(np.ceil)
LOB5_subset = LOB5.groupby("rounded_time").tail(1)
LOB5_subset.head(5)

# Walk the LOB at on the snapshot, based on Numpy
def walk_LOB(BidPrice, BidSize,Time):
    '''Walk through the buy(Bid) side, and assume the volume increase from 0 to the sum of level 5 volume N_t.  

    return a pandas dataframe of size as large as the "total volume"/50 (Buy as multiple times of 50, e.g 50,100,150...)   
    -- 'Price Impact', 'Liquidation Volume', 'Time' 
    
    Input: BidPrice (list) For LOB with 5 levels away from the midprice, it should be of length 5
           BidSize (list) The corresponding size the the 5 level.'''
    BidPrice = BidPrice.copy().values
    BidSize = BidSize.copy().values
    Price_diff = BidPrice[0] - BidPrice[1:5]   # Of length 4, best quote - worse bid, corresponding the 4 level next to the best quote
    total_volume = sum(BidSize)
    accumulated_volume = np.cumsum(BidSize) # accumulated sum
    df = []  # Initiate an empty list
    for i in range(1,int(total_volume//50*50) + 2,50):
           payment_diff = np.minimum(np.maximum(0, i - accumulated_volume[:4]),BidSize[1:5]) @ Price_diff
           shares = i
           price_impact = payment_diff/shares
           df.append([Time,shares,price_impact])
    return df
                  
# Testing
""" bp = LOB5.loc[10000,["BidPrice1","BidPrice2","BidPrice3","BidPrice4","BidPrice5"]]
bs = LOB5.loc[10000,["BidSize1","BidSize2","BidSize3","BidSize4","BidSize5"]]
t = LOB5.loc[10000,"Time"]
LOB5.loc[1][["BidPrice1","BidPrice2","BidPrice3","BidPrice4","BidPrice5"]] """
#walk_LOB(bp,bs,t)
# Test on a slice of df
""" test_sample = LOB5.loc[1:10,:]
result = test_sample.apply(lambda x: walk_LOB(x[["BidPrice1","BidPrice2","BidPrice3","BidPrice4","BidPrice5"]], x[["BidSize1","BidSize2","BidSize3","BidSize4","BidSize5"]],x["Time"]),axis=1,result_type=None).explode(ignore_index=True).to_list()
pd.DataFrame(result,columns=["Time","Volume","Price Impact"]) """

# The different volume at different snapshot, about 4e+7 rows
walk_result = LOB5_subset.apply(lambda x: walk_LOB(x[["BidPrice1","BidPrice2","BidPrice3","BidPrice4","BidPrice5"]], x[["BidSize1","BidSize2","BidSize3","BidSize4","BidSize5"]],x["Time"]),axis=1,result_type=None).explode(ignore_index=True).to_list()
walk_result = pd.DataFrame(walk_result,columns=["Time","Volume","Price Impact"])
walk_result.loc[:,"Time"] = walk_result.loc[:,"Time"].apply(np.ceil)

# Plotting the LOB plot
draw_data = walk_result.loc[(walk_result.Time<=39660)&(walk_result.Time>39600)]
# Regression
X = draw_data["Volume"].to_numpy().reshape(-1,1)
y = draw_data["Price Impact"].to_numpy().reshape(-1,1)
model = LinearRegression(fit_intercept=False)
model.fit(X,y)
slope_k = model.coef_[0][0]  # The estimated k between 11 AM and 11:01 AM


interval_point = np.arange(walk_result.Time[0]-1,walk_result.Time[37606893]+60,step=60)
walk_result["minute_interval"] =  pd.cut(walk_result.loc[:,"Time"], bins= interval_point, right=True, labels= np.arange(1,391) , retbins=False, precision=8, include_lowest=False, duplicates='raise', ordered=True)


ax = sns.lineplot(data = draw_data,x="Volume",y="Price Impact",hue = "Time",alpha = 0.4)
ax = sns.lineplot(x = np.linspace(0,1e+5,500),y = slope_k*np.linspace(0,1e+5,500),linestyle = "dashed",color="red")
ax.set_xlabel('Temporary Price Impact', fontsize = 14)
ax.set_ylabel('Volume', fontsize = 14)
plt.show()

# apply linear regresion using numpy
def linReg(x, y):
    '''linear regression using numpy starting from two one dimensional numpy arrays'''
    A = np.vstack([x]).T
    slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return pd.Series({'slope':slope})
daily_k = walk_result.groupby('minute_interval').apply(lambda x: linReg(x["Volume"], x["Price Impact"]))

daily_k.reset_index(inplace = True)

# k_oneday.png
ax = sns.lineplot(data = daily_k, x = "minute_interval",y="slope")
ax.set_xlabel('Estimated k', fontsize=14)
ax.set_ylabel('Minute', fontsize=14)
#plt.hlines(y=daily_k["slope"].mean(),xmin=0,xmax=390,colors="red",linestyles="dotted")
plt.show()

# # Estimate $\lambda +$ (hourly mean arrival)  $\lambda -$ $E(\eta +)$ $E(\eta -)$

interval_point = np.arange(np.floor(execution.Time[0]),execution.Time[581029]+1801,step=3600)
execution["hour"] = pd.cut(execution.loc[:,"Time"], bins= interval_point, right=True, labels=np.arange(1,8), retbins=False, precision=8, include_lowest=True, duplicates='raise', ordered=True)

# Average (hourly) arrival rate of MO+ MO-
df = execution[["Market Order Type","hour"]].groupby(["hour","Market Order Type"]).size().reset_index().rename({0:"Orders per hour"},axis=1)
sns.lineplot(data=df, x="hour",y="Orders per hour",hue="Market Order Type",legend=None)
sns.scatterplot(data=df, x="hour",y="Orders per hour",hue="Market Order Type")
None

execution[["Size","Market Order Type"]].groupby(["Market Order Type"]).mean()


