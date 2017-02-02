# ffn package testing
import tradingWithPython
import ffn
#%pylab inline

data = ffn.get('agg,tlt,spy,eem,efa,hyd,usmv,^GSPC', start='2010-01-01', end='2017-01-01')
print data.tail()
returns = data.pct_change()
returns.corr()
returns.plot_corr_heatmap()
ax = data.rebase().plot(figsize=(12,5))
data.calc_stats() # doesnt work...
data.tail()
data.resample('A',how='last').pct_change()

#get sharpe for each ETF
returns.mean(0)*252/( returns.std(0)*np.sqrt(252))
calculate_sharpe(data)

self.monthly_prices = obj.resample('M',how='last')
# A == year end frequency
self.yearly_prices = obj.resample('A',how='last')

plt.clf()



# ANNUAL CORRELATIONS
# calculate correlations, split by year
spx_corr = lambda x: x.corrwith(x['gspc'])
by_year = returns.groupby(lambda x: x.year)
by_year.apply(spx_corr)

#sharpe by year
by_year.apply(lambda x: x.mean()/x.std()*np.sqrt(252))



# WHAT ABOUT SSO / UBT ?
sso_ubt = ffn.get('SSO,UBT',start='2010-01-01', end='2017-01-01')
sso_ubt.tail()

sso_ubt_rets = sso_ubt.pct_change()
sso_ubt_rets.corr()
by_year = sso_ubt_rets.groupby(lambda x: x.year)
by_year.apply(lambda x: x.corrwith(x['sso']))
by_year.apply(lambda x: x.mean()/x.std()*np.sqrt(252))




