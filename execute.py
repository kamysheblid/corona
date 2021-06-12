#!/usr/bin/env python

import matplotlib.pyplot as plt
import datetime
import corona
import canada
import logging

important_regions = ['Iran', 'US', 'Canada', 'Japan',
                     'Korea, South', 'Italy', 'Spain',
                     'United Kingdom', 'Germany', 'Russia']

european_countries = ['Germany', 'United Kingdom', 'France',
                      'Spain', 'Italy', 'US', 'Canada']

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-s',action='store_true',help='Skip downloading new data and just make graphs')
parser.add_argument('-v',action='count',help='Verbose output, -vv for debug info')

args = parser.parse_args()

if args.v == 0:
    logging.basicConfig(level=logging.ERROR)
elif args.v == 1:
    logging.basicConfig(level=logging.INFO)
elif args.v == 2:
    logging.basicConfig(level=logging.DEBUG)

logging.debug(f'{args}')

if not args.s:
    corona.download_data()

logging.info('Creating data from files...')
d = corona.Data()
logging.debug('Finished creating data.')

logging.info('Initializing data...')
d.__init__()
logging.debug('Finished initializing')

today = d.deaths.columns[-1].strftime('%Y_%m_%d')
plt.rc('figure', figsize=[10, 8])

import os
datadir='data'
if not os.path.isdir(datadir):
    logging.info('Making new data folder to store graphs')
    os.mkdir(datadir)

def init_conditions():
    logging.info('Creating initial conditions plot...')
    name = f'deaths_threshold'
    d.plot_deaths_per_100k_threshold(plot=False)
    plt.savefig(f'{datadir}/deaths_percapita_threshold')
    plt.close('all')

def print_europe_info():
    logging.info('Creating europe plot...')
    d.plot_prediction_deaths_countries(european_countries,
                                             plot=False, extrapolate=30,
                                             prediction_days=10)
    plt.savefig(f'{datadir}/european')
    plt.close('all')

def print_important_regions():
    logging.info('Creating important regions plot...')
    d.plot_prediction_deaths_countries(corona.important_regions,
                                             plot=False, extrapolate=30,
                                             prediction_days=10)
    plt.savefig(f'{datadir}/imporant')
    plt.close('all')

def deaths_per100k():
    logging.info('Creating percapita deaths plot...')
    d.plot_deaths_per_100k_countries(corona.important_regions, plot=False)
    plt.savefig(f'{datadir}/deaths_per100k')
    plt.close('all')

def canada_plot():
    logging.info('Creating canadaian plot...')
    (data:=canada.canada)[data>0].dropna(axis=1,how='all').dropna(axis=0,how='all').drop('Diamond Princess').T.plot(logy=True)
    plt.ylabel('Deaths')
    plt.savefig(f'{datadir}/canada')
    plt.close('all')
    return

def canada_percapita_plot():
    logging.info('Creating canadian percapita plot...')
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

def daily_deaths(avg_days=14, logarithmic=False):
    # daily deaths are found by using a triangular rolling sum over avg_days and dividing by (avg_days-1)/2.
    # The actual covid figures are very chaotic, this makes them settle to a more visually appealing curve
    # but doesnt significantly change the data (in my opinion).
    logging.info('Creating daily deaths plot...')
    regions = ['US','Iran','Germany','Italy','Spain','United Kingdom','Brazil','India', 'Vietnam']
    ax = (dat:=(d.deaths.loc[regions].diff(axis=1).sort_values((a:=d.deaths.columns)[-1],ascending=False).get(a[1:]).rolling(avg_days,win_type='triang',axis=1).sum()/((avg_days-1)/2))).sort_values(dat.columns[-1],ascending=False).T.plot(logy=logarithmic,ylim=(1,None))
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
    logging.info('Creating weekly avg deaths plot...')
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
    deaths_per100k()
    canada_percapita_plot()
    daily_deaths()
    None
