from pandas import read_csv,date_range,DatetimeIndex,Series
import numpy as np
import matplotlib.pyplot as plt

plt.rc('axes',grid=True)
plt.rc('axes.grid',which='both')

import os
if not os.path.isfile('deaths.csv'):
    import corona
    corona.download_data()
canada = (b:=(a:=read_csv('deaths.csv')).groupby(['Country/Region','Province/State']).sum()).set_axis(DatetimeIndex(b.columns),axis=1).loc['Canada']

qc = canada.loc['Quebec']
data = qc[qc>0]


pop = {'Ontario': 14446515,'Quebec': 8433301,'British Columbia': 5020302,'Alberta': 4345737,'Manitoba': 1360396,'Saskatchewan': 1168423,'Nova Scotia': 965382,'New Brunswick': 772094,'Newfoundland and Labrador': 523790,'Prince Edward Island': 154748,'Northwest Territories': 44598,'Yukon': 40369}

per_capita=canada.copy()

for province in pop.keys():
    per_capita.loc[province] = per_capita.loc[province]/pop[province]*1e5


def plot_deaths(state=None,plot=True,thresh=1e2):
    if isinstance(state,str):
        data = (a:=canada.loc[state]
                .sort_values(canada.columns[-1],
                    ascending=False))[a>0]
    else:
        if not state:
            state = (a:=canada.sort_values(canada.columns[-1],ascending=False))[a.get(a.columns[-1])>thresh].index
        data = (b:=(a:=canada.loc[state]).T)[b>1].dropna(axis='rows',how='all')
    ax = data.plot(logy=True)
    ax.set_ylabel('Deaths')
    fig = ax.get_figure()
    if plot:
        fig.show()
    return fig

def plot_pctchange(plot=True):
    ax = (data.pct_change().dropna().plot(grid=True,title='Percent Change in Deaths',legend=True))
    fig = ax.get_figure()
    if plot: fig.show()
    return fig

def plot_deaths_daily(plot=True):
    ax = data.diff().dropna().plot(grid=True,logy=True,title='Daily Deaths')
    ax.set_ylabel('Deaths')
    fig = ax.get_figure()
    if plot: fig.show()
    return fig

def plot_death_predictions(days=70,plot=True):
    latest_deaths = data.tail(1).values
    avg_growth = data.pct_change().tail(5).mean()
    f = lambda t,a,b: a*np.exp(b*t)
    ax = (Series(data=f(np.arange(per:=100),latest_deaths,avg_growth),index=date_range(start=data.index[-1],periods=per))).head(days).plot(logy=True);
    fig = ax.get_figure()
    if plot: fig.show()
    return fig
