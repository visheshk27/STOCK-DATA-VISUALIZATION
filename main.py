import pandas as pd
import numpy as np
import yfinance
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime
plt.rcParams['figure.figsize'] = [8,5]
plt.rc('font', size=14)

lines = []
buystrength = 0

name = input("What stock would you like to scan?")
begin_time = datetime.datetime.now()
try:
  ticker = yfinance.Ticker(name)
except: 
  print("This ticker does not exist")
df = ticker.history(interval="1d",period="3mo")
df['Date'] = pd.to_datetime(df.index)
df['Date'] = df['Date'].apply(mpl_dates.date2num)
daylen = len(df['Date'].tolist())
df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

def get_current_price(symbol):
  todays_data = symbol.history(period='1d')
  return todays_data['Close'][0]
price = float(get_current_price(ticker))


def isSupport(df,i,layers,stat):
  lst = []
  stat = df[stat]
  '''
  for x in range(1,layers+1):
    lst.append(stat[i-(x-1)] < stat[i-x])
    lst.append(stat[i+(x-1)] < stat[i+x])
  '''
  for x in range(1,layers+1):
    lst.append(stat[i] < stat[i-x])
    lst.append(stat[i] < stat[i+x])
  return sum(lst) == layers*2
def isResistance(df,i,layers,stat):
  lst = []
  stat = df[stat]
  '''
  for x in range(1,layers+1):
    lst.append(stat[i-(x-1)] > stat[i-x])
    lst.append(stat[i+(x-1)] > stat[i+x])
  '''
  for x in range(1,layers+1):
    lst.append(stat[i] > stat[i-x])
    lst.append(stat[i] > stat[i+x])
  return sum(lst) == layers*2

def abline(slope,intercept):
  minim = df['Date'][0]
  maxim = max(df['Date'])
  y = intercept + slope * (61)
  plt.plot((minim, maxim), (intercept,y),('--'))

def average(lst):
  total = 0
  for l in lst:
    total += l
  return total/len(lst)

def calcLines():
  global lines
  lines_ = []
  for i in range(0,2):
    x = np.array([level[0] for level in extremes[i]], dtype = 'object').reshape(-1,1)
    y = np.array([level[1] for level in extremes[i]], dtype = 'object')
    model = LinearRegression().fit(x,y)
    coef_, intercept_ =  model.coef_,model.intercept_
    lines_.append(coef_/price)
    lines.append((coef_,intercept_))
  coef_dif = abs(lines_[0]-lines_[1])
  if abs(coef_dif) > 0.0025:
    pattern_ = "wedge"
    strength_ = int(1000*coef_dif)
    print("This stock is currently in a wedge pattern. With a wedge angle strength of " + str(strength_) + ". ")
  else:
    coef_mean = average(lines_)
    if coef_mean > 0.0005:
      pattern_ = "channel up"
      strength_ = 10000*coef_mean
    elif coef_mean < -0.0005:
      pattern_ = "channel down"
      strength_ = 10000*coef_mean
    else:
      pattern_ = "channel"
      strength_ = "0"
    print("This stock is currently in a " + pattern_ + " pattern. With a channel slope of " + str(strength_)+ ". ")

def CleanList(lst):
  lst2 = []
  for item in lst:
    if item in lst2:
      lst2.append(item)
  lst = lst2

def plot_all():
  fig, ax = plt.subplots()

  candlestick_ohlc(ax,df.values,width=0.6, \
                   colorup='green', colordown='red', alpha=0.8)


  fig.tight_layout()

  for level in levels:
    plt.hlines(level[1],xmin=df['Date'][0],\
               xmax=max(df['Date']),colors='blue')
    plt.text(df['Date'][0],level[1],str(level[2]))

  if showSwings:
    for point in points:
      plt.scatter(point[0],point[1],s=100.0,c='black')
    for i in range(0,len(points)-1):
      plt.plot((points[i][0],points[i+1][0]),(points[i][1],points[i+1][1]),'k-')
    
  #draw horizontal lines
  if prime:
    for i in range(0,2):
      abline(lines[i][0],lines[i][1])

  fig.show()

s =  np.mean(df['High'] - df['Low'])

def isFarFromLevel(l):
  lst = []
  level = 0
  least = 100000
  for x in levels:
    diff = abs(l-x[1])
    result = diff < s
    if result:
      if diff < least:
        least = diff
        level = x
      lst.append(result)
  if level != 0:
    index = levels.index(level)
    levels[index][2] += abs(s-l)/price
  return np.sum(lst) == 0

extremes = [[],[]]
levels = []
points = [(max(df['Date']),price)]
for i in range(df.shape[0]-3,df.shape[0]-daylen,-1):
    if isSupport(df,i,2,'Low'):
      l = df['Low'][i]
      extremes[0].append((i,l))
      points.append((df['Date'][i],l))
      if isFarFromLevel(l):
        levels.append([i,l,1])
        
    elif isResistance(df,i,2,'High'):
      h = df['High'][i]
      extremes[1].append((i,h))
      points.append((df['Date'][i],h))
      if isFarFromLevel(h):
        levels.append([i,h,1])

def stockVal():
  global isUptrend, buyStrength
  isUptrend = price > points[1][1]
  if len(levels) > 1:
    while len(levels) > 2:
      levels.remove(min([(l[2],l) for l in levels])[1])
    resistance = [(level[1]) for level in levels if level[1] > price]
    support = [(level[1]) for level in levels if level[1] < price]
    try:
      res = resistance[0]
      sup = support[0]
    except:
      print("{} eliminated because stock is not between S and R".format(name))
      return
    if price < res and price > sup:
      total = 0
      for l in levels:
        total += l[2]
      buyStrength = (((res+sup)/price)-2) * total * 100
      if isUptrend and buyStrength < 0:
        print("{} eliminated because stock is in up trend".format(name))
        return
      elif not isUptrend and buyStrength > 0:
        print("{} eliminated because stock is in down trend".format(name))
        return
stockVal()
  
run = True
prime = True
showSwings = True
try:
  calcLines()
except ValueError:
  print("Not prime stock")
  prime = False
print("The stock you chose is: " + name + ". The stock's current price is " + str(round(price,2)))
print("The stock's current resistance prices are:")
print([(level[1],level[2]) for level in levels if level[1] > price])
print("The stock's current support prices are:")
print([(level[1],level[2]) for level in levels if level[1] < price])
if isUptrend:
  print("The stock is currently in an uptrend.")
else:
  print("The stock is currently in a downtrend.")
print("--")
try:
  print("The stock has a BUY STRENGTH of: ", buyStrength)
except:
  print("Not ideal to buy")
print("This script took: {}".format(datetime.datetime.now() - begin_time))
if input("Plot? (y/n)") == "n":
  quit()
else:
  if input("Show Swings? (y/n)") == "n":
     showSwings = False
  plot_all()

