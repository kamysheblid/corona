#!/usr/bin/env python

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
today = d.deaths.columns[-1].strftime('%Y_%m_%d')
plt.rc('figure', figsize=[10, 8])

import os
datadir='data'
if not os.path.isdir(datadir):
    os.mkdir(datadir)

def print_europe_info():
    d.plot_prediction_deaths_countries(european_countries,
                                             plot=False, extrapolate=30,
                                             prediction_days=10)
    plt.savefig(f'{datadir}/european')
    plt.close('all')

def print_important_regions():
    d.plot_prediction_deaths_countries(corona.important_regions,
                                             plot=False, extrapolate=30,
                                             prediction_days=10)
    plt.savefig(f'{datadir}/imporant')
    plt.close('all')

def init_conditions():
    name = f'deaths_threshold'
    d.plot_deaths_from_threshold(plot=False)
    plt.savefig(f'{datadir}/deaths_threshold')
    plt.close('all')

def deaths_per100k():
    d.plot_deaths_per_100k_countries(corona.important_regions, plot=False)
    plt.savefig(f'{datadir}/deaths_per100k')
    plt.close('all')

def canada_plot():
    (data:=canada.canada)[data>0].dropna(axis=1,how='all').dropna(axis=0,how='all').drop('Diamond Princess').T.plot(logy=True)
    plt.ylabel('Deaths')
    plt.savefig(f'{datadir}/canada')
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
    plt.savefig(f'{datadir}/canada_percapita')
    plt.close('all')
    return


def print_canada():
    (d.raw_deaths.groupby(['Country/Region','Province/State']).sum().set_axis(d.deaths.columns,axis=1).loc['Canada'].diff(axis=1).sort_values(d.deaths.columns[-1],ascending=False).T.rolling(7).sum()/7).plot(logy=False)
    plt.title('Canada Daily Deaths')
    plt.ylabel('Deaths')
    plt.savefig(f'{datadir}/canada_dailydeaths')
    plt.close('all')

def daily_deaths(avg_days=14):
    # daily deaths are found by using a triangular rolling sum over avg_days and dividing by (avg_days-1)/2.
    # The actual covid figures are very chaotic, this makes them settle to a more visually appealing curve
    # but doesnt significantly change the data (in my opinion).
    regions = ['US','Iran','Germany','Italy','Spain','United Kingdom','Brazil','India']
    ax = (dat:=(d.deaths.loc[regions].diff(axis=1).sort_values((a:=d.deaths.columns)[-1],ascending=False).get(a[1:]).rolling(avg_days,win_type='triang',axis=1).sum()/((avg_days-1)/2))).sort_values(dat.columns[-1],ascending=False).T.plot(logy=False,ylim=(1,None))
    plt.title('Daily Deaths')
    plt.ylabel('Deaths')
    plt.xlabel('Date')
    fig = ax.get_figure()
    fig.savefig(f'{datadir}/daily_deaths')
    plt.close('all')
    return

def daily_avg_by_week_deaths():
    # Finds the avg daily death by summing over a week and dividing by 7.
    # This is a more stable representation of deaths
    ax = d.deaths_weekly.T.plot(logy=False)
    plt.title('Weekly Deaths')
    plt.ylabel('Deaths')
    plt.xlabel('Date')
    fig = ax.get_figure()
    fig.savefig(f'{datadir}/weekly_deaths')
    plt.close('all')
    return

if __name__ == '__main__':
    init_conditions()
    print_europe_info()
    print_important_regions()
    deaths_per100k()
    canada_percapita_plot()
    daily_deaths()
    print_canada()
    None
