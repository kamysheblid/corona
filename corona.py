import os
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

confirmed_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
recovered_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
deaths_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
population_url = 'https://raw.githubusercontent.com/datasets/population/master/data/population.csv'

# worldbank took down their api sometime in Jun
#population_url = ("http://api.worldbank.org/countries/all/indicators/SP.POP.TOTL?format=csv")

important_regions = [
    "Iran",
    "US",
    "Canada",
    "China",
    "Japan",
    "Korea, South",
    "Italy",
    "Spain",
    "Russia",
    "United Kingdom",
    "Brazil",
    "Germany",
    "India",
]

plt.rc('axes',grid=True)
plt.rc('axes.grid',which='both')

class Data:
    def __init__(
        self,
        confirmed_file="confirmed.csv",
        deaths_file="deaths.csv",
        recovered_file="recovered.csv",
    ):
        """files can be url or filename"""

        import os
        if not all([os.path.isfile(a) for a in [confirmed_file,deaths_file,recovered_file]]):
            download_data()

        self.raw_confirmed = pd.read_csv(confirmed_file)

        self.country_names = (
            self.raw_confirmed.get("Country/Region").drop_duplicates().values
        )

        self.population = self._get_population_figures()

        self.confirmed = self.raw_confirmed.groupby(by="Country/Region").sum()
        # fix dates
        self.confirmed.columns = pd.date_range(
            start=(a := self.confirmed.columns)[0], periods=a.size
        )

        self.affected_confirmed = self._by_countries_affected(data=self.confirmed)
        self.affected_confirmed_xchina = self.affected_confirmed.drop("China")

        # self.confirmed_per_100k = self.raw_confirmed.groupby('Country/Region').sum().divide(self.population,axis=0).dropna().mul(1e5).sort_values(self.raw_confirmed.columns[-1],ascending=False)

        self.raw_deaths = pd.read_csv(deaths_file)
        self.deaths = self.raw_deaths.groupby(by="Country/Region").sum()
        # Fix dates
        self.deaths.columns = pd.date_range(
            start=(a := self.deaths.columns)[0], periods=a.size
        )
        self.newest_day = self.deaths.columns[-1]

        self.confirmed_per100k = (
            self.confirmed.divide(self.population, axis=0)
            .dropna(how="all")
            .mul(1e5)
            .sort_values(self.confirmed.columns[-1], ascending=False)
        )

        self.deaths_per_100k = (
            self.deaths.divide(self.population, axis=0)
            .dropna()
            .mul(1e5)
            .sort_values(self.deaths.columns[-1], ascending=False)
        )
        self.deaths_daily = (
            self.deaths.diff(axis=1)
            .dropna(axis=1)
            .sort_values(self.deaths.columns[-2], ascending=False)
        )
        self.deaths_weekly = (
                (a:=self.deaths_daily.T.resample('W').sum().T)
            .sort_values(a.columns[-2],ascending=False)
        )

        self.growth = self._confirmed_growth_rates()

        self.raw_recovered = pd.read_csv(recovered_file)
        self.recovered = self.raw_recovered.groupby("Country/Region").sum()

        self.recovered_confirmed_ratio = (
            (a := self.recovered)
            .div(self.confirmed)
            .mul(100, fill_value=0)
            .sort_values(a.columns[-1], ascending=False)
        )


        return None

    def _plot_major_deaths_daily(self):
        (
            a := (s := self.deaths)
            .sort_values(s.columns[-1], ascending=False)
            .head(20)
            .diff(axis="columns")
            .sort_values(s.columns[-1], ascending=False)
        )[a > 20].T.plot()
        plt.show()
        return None

    def _plot_major_confirmed_daily(self):
        (
            a := (s := self.confirmed)
            .sort_values(s.columns[-1], ascending=False)
            .head(20)
            .diff(axis="columns")
            .sort_values(s.columns[-1], ascending=False)
        )[a > 20].T.plot()
        plt.show()
        return None

    def _load_data(self):
        self.__init__()

    def _download_data(self):
        sites = dict(
            [
                ["confirmed", confirmed_url],
                ["deaths", deaths_url],
                ["recovered", recovered_url],
            ]
        )
        for file in ["confirmed", "deaths", "recovered"]:
            expr_read = f'pd.read_csv({file}_url).drop(["Lat","Long"],axis="columns")'

            print(expr_read)
            local_expr = dict([["memory", eval(expr_read)]])
            local_vars = dict(**sites, **local_expr)

            # expr_write = "memory.to_csv('{0}.csv',index=None)".format(file)
            expr_write = f'memory.to_csv("{file}.csv",index=None)'
            print(expr_write)
            eval(expr_write, local_vars)
        return None

    def _download_data2(self):
        for kind in ["confirmed", "deaths", "recovered"]:

            raw = (
                pd.read_csv(
                    f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{kind}_global.csv"
                )
                .drop(["Lat", "Long"], axis="columns")
                .groupby("Country/Region")
                .sum()
                .rename(columns=pd.to_datetime)
            )

            raw.to_csv(f"{kind}.csv")
        return None

    def _death_growth_rate(self, country):
        data = (data := self.deaths.loc[country])[data > 0]
        return self.fitting(data)

    def _graph_growth_rate(self, country, days=30, plot=True):
        data = (data := self.deaths.loc[country])[data > 0]
        dates = pd.date_range(start=data.index[0], periods=days)

        a, b = self.fitting(data)

        f = lambda x: a * np.exp(x * b)
        x = np.arange(days)

        pd.Series(f(x), index=dates).plot()
        if plot:
            plt.show()

    def fitting(self, ydata):
        f = lambda t, a, b: a * np.exp(b * t)
        popt, pcov = curve_fit(f, ydata=ydata, xdata=np.arange(ydata.size))
        return popt

    def _prediction_confirmed(self, country, days=30, plot=True, logy=True):
        country_confirmed = pd.Series(
            (b := (a := self.confirmed.loc[country])[a > 10]),
            index=pd.date_range(start=b.index[0], periods=b.size),
        )
        dates_confirmed = country_confirmed.index

        a, b = self.fitting(country_confirmed)

        confirmed_equation = lambda days: a * np.exp(days * b)
        xdata = np.arange(days + country_confirmed.size)
        prediction_confirmed = confirmed_equation(xdata)

        prediction_dates = pd.date_range(
            start=dates_confirmed[0], periods=prediction_confirmed.size
        )

        prediction_confirmed = pd.Series(
            confirmed_equation(xdata), index=prediction_dates
        )

        prediction_confirmed.plot(logy=logy)
        country_confirmed.plot(logy=logy)

        if plot:
            plt.show()
        return None

    def plot_prediction_deaths_countries(
        self,
        regions: iter,
        prediction_days: int = 7,
        extrapolate: int = 45,
        logy: bool = True,
        plot=True,
        xgrid=None,
        ygrid=None,
        title="Deaths Extrapolation",
    ):

        data = self.deaths.loc[regions].sort_values(self.deaths.columns[-1], ascending=False)
        regions = data.index

        for country in regions:
            fig = self._prediction(
                data=self.deaths,
                country=country,
                prediction_days=prediction_days,
                plot=False,
                logy=logy,
                extrapolate=extrapolate,
                xgrid=xgrid,
                ygrid=ygrid,
                title="Deaths Extrapolation",
            )
        plt.legend()
        plt.ylabel("Deaths Extrapolation")
        plt.xlabel("Date")
        plt.grid(axis="x", which=xgrid) if xgrid else None
        plt.grid(axis="y", which=ygrid) if ygrid else None
        plt.title(title)

        for ax in fig:
            ax.set_autoscaley_on(False)
            ax.vlines(self.deaths.columns[-1],*ax.get_ylim(),color='r',linestyle='--')
            #[ax.legendlabels.remove(label) for label in ax.legendlabels if 'prediction' in label]

        if plot:
            plt.show()
        return fig

    def _prediction(
        self,
        data,
        country: str,
        extrapolate: int = 30,
        prediction_days: int = 10,
        plot: bool = True,
        logy: bool = True,
        xgrid="both",
        ygrid="both",
        title="",
    ):
        country_data = data.loc[country]
        a, b = self.fitting(country_data.tail(prediction_days))

        dates = country_data.tail(prediction_days).index

        equation = lambda days: a * np.exp(b * days)
        xdata = np.arange(extrapolate + prediction_days)
        prediction_data = equation(xdata)

        prediction_dates = pd.date_range(
            start=dates[0], periods=prediction_days + extrapolate
        )
        prediction_data = pd.Series(
            prediction_data,
            index=prediction_dates,
        #    name="%s prediction" % country_data.name,
        )

        fig1 = prediction_data.plot(logy=logy)
        fig2 = country_data.plot(logy=logy)
        if plot:
            plt.show()
        return [fig1, fig2]

    def _prediction_deaths(
        self, countries: iter, prediction_days=10, future_days=30, plot=True, logy=True
    ):
        fig = []
        for country in countries:
            country_deaths = pd.Series(
                (b := (a := self.deaths.loc[country])[a > 10]),
                index=pd.date_range(start=b.index[0], periods=b.size),
            )

            dates_deaths = country_deaths.index
            a, b = self.fitting(country_deaths.tail(prediction_days))

            deaths_equation = lambda days: a * np.exp(b * days)
            xdata = np.arange(future_days + country_deaths.size)
            prediction_deaths = deaths_equation(xdata)

            prediction_dates = pd.date_range(
                start=dates_deaths[0], periods=prediction_deaths.size
            )
            prediction_deaths = pd.Series(
                prediction_deaths, index=prediction_dates, name=f"{country} prediction"
            )

            ax1 = prediction_deaths.plot(logy=logy)
            ax2 = country_deaths.plot(logy=logy)
            for i in [ax1, ax2]:
                fig.append(i.get_figure())
        plt.legend()
        #plt.grid(b=True, which="major", axis="y")
        #plt.grid(b=True, which="both", axis="x")
        if plot:
            plt.show()
        if save:
            plt.savefig(save)
        return fig

    def _prediction_(
        self, countries, days=10, future=30, threshold=1e1, plot=True, logy=True
    ):
        final_data = pd.DataFrame()
        for country in countries:
            country_deaths = (a := self.deaths.loc[country])
            data = country_deaths[country_deaths > threshold].tail(days)
            a, b = data.iloc[0], self.fitting(data)[1]

            deaths_equation = lambda d: a * np.exp(d * b)

            import datetime

            delta = datetime.timedelta(days=int(np.log(a) / b))
            day1 = data.index[0] - delta

            xdata = np.arange(-delta.days, future)
            prediction_deaths = deaths_equation(xdata)

            prediction_dates = pd.date_range(start=day1, periods=xdata.size)

            prediction_deaths = pd.Series(
                prediction_deaths, index=prediction_dates, name=f"{country} prediction"
            )

            prediction_deaths[
                prediction_deaths.index
                > (a := self.deaths.loc[country])[a > 0].index[0]
            ].plot(logy=logy)
            country_deaths.plot(logy=logy)
            final_data[country] = prediction_deaths
        # if plot:
        # final_data[final_data.index>'2020-02-01'].plot(logy=logy)
        plt.legend()
        plt.title("Covid Death Predictions")
        plt.ylabel("Deaths" if not logy else "log(Deaths)")
        plt.xlabel("Date")
        #plt.grid(b=True, which="major", axis="y")
        #plt.grid(b=True, which="both", axis="x")
        plt.show()
        return final_data

    def _confirmed_growth_rates(self, days=30):
        dates = self.confirmed.columns
        data = self.confirmed.sort_values(dates[-1], ascending=False)

        # Without doing next part ,some countries fail curve_fit since figures too low
        countries = data.index[data.get(dates[-1]) > 15]

        f = lambda t, a, b: a * np.exp(b * t)
        a = pd.Series(dtype="float64")
        res = 0

        from scipy.optimize import OptimizeWarning

        for country in countries:
            country_days = days
            # If country has too many 0s then skip it
            if (c := data.loc[country])[c.index[-3]] == 0:
                continue

            while True:
                try:
                    ydata = data.loc[country].tail(country_days)
                    res = curve_fit(f, ydata=ydata, xdata=np.arange(ydata.size))[0][1]
                    break
                except OptimizeWarning as e:
                    print(e)
                    print("Increasing days", country_days)
                    country_days += 1
                except Exception as e:
                    e.args += (res,)
                    print("Unknown Error")
                    raise
                break
            s = pd.Series(res, index=[country])
            a = a.append(s)
        return a.sort_values(ascending=False).mul(100)

    def print_growth_rates(days=30):
        self._confirmed_growth_rates(days)
        return None

    def _pct_growth(self, country, days=30, threshold=0.2):
        confirmed_pct = (
            c := (a := self.confirmed)
            .groupby("Country/Region")
            .sum()
            .loc[country]
            .get(a.columns[-days:])
            .pct_change()
        ).where(c > threshold, c < 1)

        deaths_pct = (
            c := (a := self.deaths)
            .groupby("Country/Region")
            .sum()
            .loc[country]
            .get(a.columns[-days:])
            .pct_change()
        ).dropna()
        return None

    def country_info(self, country):
        """country and to_screen (whether to print). 
If no print then return confirmed,deaths"""
        confirmed = self.confirmed.loc[country]
        deaths = self.deaths.loc[country]

        if country in self.country_names:
            return pd.DataFrame(
                [confirmed.rename("confirmed"), deaths.rename("deaths")]
            )
        else:
            raise ValueError("%s is not a part of the data" % (country))

    def plot_growth_rate(self, days=30):
        days = days
        self._confirmed_growth_rates(days=days).T.plot(kind="bar")
        plt.show()
        return None

    def _per100k_growth_rates(self, days=30, failure=False):
        """failure notifies if any of countries failed in the fit"""
        dates = self.per_100k.columns
        data = self.per_100k.sort_values(dates[-1], ascending=False)
        # .drop(dates[-1],axis='columns')

        arr = pd.Series(dtype="float64")
        for country in data.index:
            country_data = data.loc[country][-days:]

            # If the last 2 values are equal get rid of them to avoid
            # wrong calculation
            if country_data[-2:].duplicated().any():
                country_data.drop(dates[-1], inplace=True)

            # if the data of the country is more than 90% 0 then do
            # not fit or it will crash system
            # if country_data.value_counts().loc[0] > 0.9 * country_data.size:
            # continue
            try:
                popt = self.fitting(country_data)
            except:
                continue
            s = pd.Series(popt[1], index=[country])
            arr = arr.append(s)

        if arr.index.size != data.index.size and failure:
            print(
                "Could not find values for",
                [x for x in self.per_100k.index if x not in arr.index],
            )
        return arr.sort_values().mul(100)

    def _how_many_are_sick_today(self, country):
        date_latest = self.confirmed.columns[-1]
        return self.confirmed.loc[country].get(date_latest)

    def _write_to_file(self):
        c = "Country/Region"
        a = dict(
            [
                ["confirmed", [False, None]],
                ["population", [True, c]],
                ["countries_confirmed", [True, c]],
                ["affected", [True, c]],
                ["per_100k", [True, c]],
                ["deaths", [True, c]],
                ["deaths_per_100k", [True, c]],
                ["deaths_daily", [True, c]],
            ]
        )
        for file in a.keys():
            bool_val = a[file][0]
            country = a[file][1]
            if bool_val:
                eval(
                    "self.{0}.to_csv('{0}.csv',index={1},index_label='{2}')".format(
                        file, *a[file]
                    )
                )
            else:
                eval("self.{0}.to_csv('{0}.csv',index=False)".format(file))
        return None

    def _get_population_figures(self):
        pop_data = (pop:=pd.read_csv(population_url))[pop.Year==2018].drop(['Country Code','Year'],axis=1)

        #pop_data = pop[pop.Year==2018].drop(['Country Code','Year'],axis=1)


        pop = pd.Series(data=pop_data.Value.values,index=pop_data['Country Name'].values)

        def search_word(word,l1): 
            return [a for a in l1 if word in a or a in word]

        renaming_dict = dict([(search_word(c,pop.index.values)[0],c) for c in [a for a in self.country_names if a not in pop.index.values] if search_word(c,pop.index.values) != []] + [
            ("Congo (Brazzaville)","Congo, Rep."),
            ("Congo (Kinshasa)", "Congo, Dem. Rep."),
            ("Myanmar", "Burma"),
            ("Czech Republic", "Czechia"),
            ("Korea, Rep.", "Korea, South"),
            ("Kyrgyz Republic", "Kyrgyztan"),
            ("Lao PDR", "Laos"),
            ("St. Kitts and Nevis", "Saint Kitts and Nevis"),
            ("St. Lucia", "Saint Lucia"),
            ("St. Vincent and Grenadines", "Saint Vincent and the Grenadines"),
            ("Slovak Republic", "Slovakia"),
            ("United States", "US")])

        return pop.rename(renaming_dict)
 
    def _per_capita(self, data):
        arr = pd.DataFrame()

        # Latest year with data in self.population
        for country in data.index:
            for date in data.columns:
                sick = data.get_value(index=country, col=date)
                pop = self.population[country]

                per_capita = sick / pop * 1e5

                arr.set_value(index=country, col=date, value=per_capita)
        return arr

    def _by_countries_affected(self, data, threshold=1e3):
        newest_data = data.get(data.columns[-1])
        unaffected = newest_data < threshold

        for drop_country in unaffected.keys():
            if unaffected[drop_country]:
                data = data.drop(drop_country)
        return data

    def print_confirmed(self):
        d = self.confirmed.to_dict("indx")
        for countries in d:
            print(countries, d[countries])
        return None

    def print_affected(self):
        d = self.affected_confirmed.to_dict("indx")
        for countries in d:
            print(countries, d[countries])
        return None

    def plot_confirmed(self, plot=True):
        self._graph(self.confirmed, plot=plot)
        return None

    def plot_confirmed_affected(self, threshold=1e2, plot=True):
        data = (
            a := (b := self.confirmed).sort_values(
                date := b.columns[-1], ascending=False
            )
        )[a.get(date) > threshold]
        self._graph(data, plot)
        return None

    def plot_confirmed_xchina(self, plot=True):
        self._graph(self.affected_confirmed_xchina, plot)
        return None

    def plot_confirmed_per100k(self, countries=None, plot=True, logy=True):
        if countries:
            data = self.confirmed_per100k.loc[countries].sort_values(
                self.confirmed.columns[-1], ascending=False
            )
        else:
            data = self.confirmed_per100k.sort_values(
                self.confirmed.columns[-1], ascending=False
            )
        ax = data.T.plot(logy=logy)
        plt.ylabel("infections per 100k")
        #plt.grid(axis='both',which='both')

        # ax = self._graph(data=data,
        #                plot=plot,
        #             ylabel='infections per 100k',
        #            logy=True)

        if plot:
            plt.show()
        return ax.get_figure()

    def plot_deaths(self, plot=True):
        # dates = self.deaths.columns[:-1]
        # data = self.deaths.drop(dates[-1],axis='columns')
        dates = self.deaths.columns
        data = self.deaths.sort_values(dates[-1], ascending=False)

        self._graph(data=data, plot=plot, ylabel="Deaths", logy=False)
        return None

    def plot_deaths_countries(
        self, regions: iter, logy: bool = True, plot: bool = True
    ):
        self._plot_countries(
            data=self.deaths, regions=regions, logy=logy, ylabel="Deaths"
        )
        if plot:
            plt.show()
        return None

    def plot_deaths_daily(
        self, regions: iter = important_regions, logy: bool = True, plot: bool = True
    ):
        ax = (
            self.deaths.diff(axis="columns")
            .loc[regions]
            .sort_values(self.deaths.columns[-1], ascending=False)
            .T.plot(logy=logy)
        )

        #plt.grid(axis="both", which="both")
        if plot:
            plt.show()
        return ax

    def plot_deaths_weekly(
            self, regions: iter = important_regions, logy: bool = True, plot: bool = True
    ):
        ax = (
            self.deaths_weekly
            .loc[[a for a in self.deaths_weekly.index if a in regions]]
            .T.plot(logy=logy)
        )

        #plt.grid(axis="both", which="both")
        if plot:
            plt.show()
        return ax

    def plot_deaths_per_100k_countries(
        self, regions: iter, logy: bool = True, plot: bool = True
    ):
        data = self.deaths_per_100k.loc[regions].sort_values(
            self.deaths.columns[-1], axis=0, ascending=False
        )


        fig = data.T.plot(logy=logy)
        plt.title('Deathrate')
        plt.ylabel('Deathrate per 100k')
        #plt.grid(axis='both',which='both')


        #fig = plt.figure()
        #self._plot_countries(
            #data=data,
            #regions=data.index,
            #logy=logy,
            #ylabel="Deathrate per 100k",
            #title="Deathrate",
            #plot=False,
        #)
        if plot:
            plt.show()
        return fig.get_figure()

    def plot_deaths_per_100k(self, plot=True):
        data = (a := self.deaths_per_100k).sort_values(a.columns[-1], ascending=False)
        self._graph(data=data, plot=plot, ylabel="Deaths per 100k",xlabel="Date", logy=False)
        if plot:
            plt.show()
        return None

    def plot_confirmed_countries(
        self, regions: iter, logy: bool = True, plot: bool = True
    ):
        data = self.confirmed.loc[regions].sort_values(
            self.confirmed.columns[-1], axis=0, ascending=False
        )
        regions = data.index

        self._plot_countries(
            data=data, regions=regions, logy=logy, ylabel="Confirmed", plot=plot,
        )
        return None

    def plot_deaths_from_threshold(
        self,
        regions: iter = important_regions,
        threshold: int = 1e2,
        logy: bool = True,
        plot=True,
        indexline=False,
    ):

        data = self.deaths

        regions = (
            (a := data.loc[regions])[a.get(a.columns[-1]) > threshold]
            .sort_values(data.columns[-1], ascending=False)
            .index
        )

        if indexline:
            f_indx = lambda t: threshold * np.exp(0.3 * t)
            xdata = self.deaths.columns
            ydata = f_indx(np.arange(xdata.size))
            index_data = pd.Series(data=ydata, index=xdata)
            data.loc["index"] = index_data[index_data < data.max().max()]
            regions = regions.append(pd.Index(["index"]))

        from functools import partial

        graph = partial(
            self._graph,
            plot=False,
            ylabel="Deaths",
            xlabel=f"Days since {int(threshold)} deaths",
            xgrid_which='both',
            ygrid_which='both',
            logy=logy,
            title=f"Effectiveness of Response since {int(threshold)} deaths",
        )

        for country in regions:
            country_data = (
                (b := (a := data.loc[country])[a > threshold])
                .set_axis(np.arange(b.size))
                .T
            )
            fig = graph(data=country_data)
        #            fig = self._graph(data=country_data,
        #                              plot=False,
        #                              ylabel='Deaths',
        #                              logy=logy,
        #                              xgrid_which='major',
        #                              ygrid_which='both',
        #                              title='Effectiveness of Response since 100th death')

        #        country_data = pd.DataFrame(index=regions,columns=np.arange(1000))
        #        for country in regions:
        #            country_data.loc[country] = (b:=(a:=data.loc[country])[a>threshold]).set_axis(np.arange(b.size))

        #        fig = country_data.T.plot(logy=logy)
        #        plt.ylabel('Deaths')
        #        #plt.grid(axis='x',which='major')
        #        #plt.grid(axis='y',which='both')
        #        plt.title(f'Effectiveness of Response since {int(threshold)} death')
        #        plt.xlabel(f'Days since {int(threshold)} deaths')
        #plt.grid(axis='y',which='both')
        if plot:
            plt.show()
        return fig

    def plot_deaths_per_100k_threshold(self,
                                       countries=important_regions,
                                       logy=True, threshold=0.01,
                                       plot=True):
        data = (d.deaths_per_100k.loc[countries].sort_values(d.deaths.columns[-1],ascending=False))
        new_data = pd.DataFrame()

        for country in data.index:
            country_data = (data
                            .loc[country][data.loc[country] > threshold]
                            .values)
            series = pd.Series(country_data,
                               name=country)
            new_data = new_data.append(series)

        ax = new_data.T.plot(logy=logy)
        ax.set_xlabel(f'Days since {threshold} deaths per 100k population')
        ax.set_ylabel('Deaths per 100k population')
        fig = ax.get_figure()
        if plot:
            fig.show()
        return fig

    def plot_confirmed_from_threshold(
        self,
        regions: iter = important_regions,
        threshold: int = 1e2,
        logy: bool = True,
        plot=True,
    ):
        data = self.confirmed

        regions = (
            (a := data.loc[regions])[a.get(a.columns[-1]) > threshold]
            .sort_values(data.columns[-1], ascending=False)
            .index
        )
        for country in regions:
            country_data = (
                (b := (a := data.loc[country])[a > threshold])
                .set_axis(np.arange(b.size))
                .T
            )
            self._graph(data=country_data, plot=False, ylabel="Confirmed", logy=logy)
        if plot:
            plt.xlabel(f"Days since {int(threshold)} confirmed")
            plt.show()
        return None

    def _plot_countries(
        self,
        data,
        regions: iter,
        logy: bool = True,
        plot: bool = False,
        ylabel: str = "",
        title="",
        xgrid="both",
        ygrid="both",
    ):
        if isinstance(regions,str):
            country_data = data.loc[regions]
            fig = self._graph(
                data=country_data.T,
                plot=plot,
                logy=logy,
                ylabel=ylabel,
                title=title,
                xgrid_which=xgrid,
                ygrid_which=ygrid,
            )
        else:
            for country in regions:
                country_data = data.loc[country]
                fig = self._graph(
                    data=country_data.T,
                    plot=False,
                    logy=logy,
                    ylabel=ylabel,
                    title=title,
                    xgrid_which=xgrid,
                    ygrid_which=ygrid,
                )
            if plot:
                fig.show()
        return fig

    def _graph(
        self,
        data,
        plot,
        ylabel="Infected",
        xlabel="Date",
        total=None,
        logy=True,
        fig=None,
        title="",
        xgrid_which="both",
        ygrid_which="both",
    ):
        # if logy:
        # ylabel = "log({0:s})".format(ylabel)
        import matplotlib.pyplot as plt

        if data.ndim == 2:
            ax = data.sort_values(data.columns[-1], ascending=False).T.plot(logy=logy)
        elif data.ndim == 1:
            ax = data.T.plot(logy=logy)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.grid(which=xgrid_which, axis="x")
        #plt.grid(which=ygrid_which, axis="y")
        plt.legend(loc="upper left", fontsize="x-small")
        if plot:
            plt.show()
        return ax.get_figure()


if __name__ == "__main__":
    d = Data()


def print_graphs():
    import datetime

    today = datetime.date.today()

    d = Data()
    european_countries = [
        "Germany",
        "United Kingdom",
        "France",
        "Spain",
        "Italy",
        "US",
        "Canada",
    ]
    d.plot_prediction_deaths_countries(european_countries, plot=False)

    plt.savefig(
        "european_predictions_{0:s}.png".format(today.strftime("%b_%d")), format="png"
    )

    return None


def download_data():
    sites = dict(
          [
                ["confirmed", confirmed_url],
                ["deaths", deaths_url],
                ["recovered", recovered_url],
            ]
        )
    for file in ["confirmed", "deaths", "recovered"]:
        expr_read = f'pd.read_csv({file}_url).drop(["Lat","Long"],axis="columns")'

        print(expr_read)
        local_expr = dict([["memory", eval(expr_read)]])
        local_vars = dict(**sites, **local_expr)

        expr_write = f'memory.to_csv("{file}.csv",index=None)'
        print(expr_write)
        eval(expr_write, local_vars)


def main():
    return None
