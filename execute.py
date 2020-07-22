import matplotlib.pyplot as plt
import datetime
import corona
import canada

important_regions = ['Iran', 'US', 'Canada', 'Japan',
                     'Korea, South', 'Italy', 'Spain',
                     'United Kingdom', 'Germany', 'Russia']

european_countries = ['Germany', 'United Kingdom', 'France',
                      'Spain', 'Italy', 'US', 'Canada']

corona.download_data()
d = corona.Data()
d.__init__()
today = d.deaths.columns[-1].strftime('%m_%d')
plt.rc('figure', figsize=[10, 8])

import os
datadir='data'
if not os.path.isdir(datadir):
    os.mkdir(datadir)

def print_europe_info():
    d.plot_prediction_deaths_countries(european_countries,
                                             plot=False, extrapolate=30,
                                             prediction_days=10)
    plt.savefig(f'{datadir}/european_{today}')
    plt.close('all')

def print_important_regions():
    d.plot_prediction_deaths_countries(corona.important_regions,
                                             plot=False, extrapolate=30,
                                             prediction_days=10)
    plt.savefig(f'{datadir}/imporant_{today}')
    plt.close('all')

def init_conditions():
    name = f'deaths_threshold_{today}'
    d.plot_deaths_from_threshold(plot=False)
    plt.savefig(f'{datadir}/deaths_threshold_{today}')
    plt.close('all')

def deaths_per100k():
    d.plot_deaths_per_100k_countries(corona.important_regions, plot=False)
    plt.savefig(f'{datadir}/deaths_per100k_{today}')
    plt.close('all')

def canada_plot():
    (data:=canada.canada)[data>0].dropna(axis=1,how='all').dropna(axis=0,how='all').drop('Diamond Princess').T.plot(logy=True)
    #plt.grid(axis='both',which='both')
    plt.ylabel('Deaths')
    plt.savefig(f'{datadir}/canada_{today}')
    plt.close('all')
    return

def canada_percapita_plot():
    provinces = [i for i in canada.per_capita.sort_values(canada.per_capita.columns[-1],ascending=False).index]
    ((b:=(a:=canada.per_capita)[a>0]
        .dropna(axis=1,how='all')
        .dropna(axis=0,how='all')
        .drop('Diamond Princess'))
            .loc[[prov for prov in b.sort_values(b.columns[-1],ascending=False).index]]
            .T.plot(logy=True))
    #plt.grid(axis='both',which='both')
    plt.ylabel('Deaths per 100k population')
    plt.savefig(f'{datadir}/canada_percapita_{today}')
    plt.close('all')
    return


def print_canada():
    d.raw_deaths.groupby(['Country/Region','Province/State']).sum().set_axis(d.deaths.columns,axis=1).loc['Canada'].T.plot(logy=True)
    plt.title('Canada Deaths')
    plt.ylabel('Deaths')
    #plt.grid('both','both')
    plt.savefig(f'{datadir}/canada_deaths_{today}')
    plt.close('all')

def daily_deaths(avg_days=6):
    regions = ['US','Iran','Germany','Italy','Spain','United Kingdom','Brazil','India']
    ax = (d.deaths.loc[regions].diff(axis=1).sort_values((a:=d.deaths.columns)[-1],ascending=False).get(a[1:]).rolling(avg_days,win_type='triang',axis=1).sum()/(avg_days/2)).T.plot(logy=True,ylim=(1,None))
    plt.title('Daily Deaths')
    plt.ylabel('Deaths')
    plt.xlabel('Date')
    fig = ax.get_figure()
    fig.savefig(f'{datadir}/daily_deaths_{today}')
    plt.close('all')
    return

def weekly_deaths():
    ax = d.deaths_weekly.T.plot(logy=False)
    plt.title('Weekly Deaths')
    plt.ylabel('Deaths')
    plt.xlabel('Date')
    fig = ax.get_figure()
    fig.savefig(f'{datadir}/weekly_deaths_{today}')
    plt.close('all')
    return

if __name__ == '__main__':
    print_europe_info()
    print_important_regions()
    init_conditions()
    deaths_per100k()
    canada_percapita_plot()
    daily_deaths()
    None
