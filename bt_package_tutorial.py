# bt package tutorial
# http://pmorissette.github.io/bt/
import bt
import pandas as pd 
# fetch some data
data = bt.get('spy,agg', start='2010-01-01')
print data.head()

# create the strategy
s = bt.Strategy('s1', [bt.algos.RunMonthly(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighEqually(),
                       bt.algos.Rebalance()])
                       
# create a backtest and run it
test = bt.Backtest(s, data)
res = bt.run(test)

