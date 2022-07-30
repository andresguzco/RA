import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from stargazer.stargazer import Stargazer
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.graphics.regressionplots import plot_leverage_resid2
import seaborn as sns
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Type


style_talk = 'seaborn-talk'  # refer to plt.style.available


class Linear_Reg_Diagnostic():
    """
    Diagnostic plots to identify potential problems in a linear regression fit.
    Mainly,
        a. non-linearity of data
        b. Correlation of error terms
        c. non-constant variance
        d. outliers
        e. high-leverage points
        f. collinearity

    Author:
        Prajwal Kafle (p33ajkafle@gmail.com, where 3 = r)
        Does not come with any sort of warranty.
        Please test the code one your end before using.
    """

    def __init__(self, results):

        self.results = maybe_unwrap_results(results)

        self.y_true = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        self.residual = np.array(self.results.resid)
        influence = self.results.get_influence()
        self.residual_norm = influence.resid_studentized_internal
        self.leverage = influence.hat_matrix_diag
        self.cooks_distance = influence.cooks_distance[0]
        self.nparams = len(self.results.params)

    def __call__(self, plot_context='seaborn-paper'):
        # print(plt.style.available)
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
            self.residual_plot(ax=ax[0, 0])
            self.qq_plot(ax=ax[0, 1])
            self.scale_location_plot(ax=ax[1, 0])
            self.leverage_plot(ax=ax[1, 1])
            plt.show()

        self.vif_table()
        return fig, ax

    def residual_plot(self, ax=None):
        """
        Residual vs Fitted Plot

        Graphical tool to identify non-linearity.
        (Roughly) Horizontal red line is an indicator that the residual has a linear pattern
        """
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.sort(residual_abs))
        abs_resid_top_3 = abs_resid[:3]
        for i, _ in enumerate(abs_resid_top_3):
            ax.annotate(
                i,
                xy=(self.y_predict[i], self.residual[i]),
                color='C3')

        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax

    def qq_plot(self, ax=None):
        """
        Standarized Residual vs Theoretical Quantile plot

        Used to visually check if residuals are normally distributed.
        Points spread along the diagonal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            ax.annotate(
                i,
                xy=(np.flip(QQ.theoretical_quantiles, 0)[r], self.residual_norm[i]),
                ha='right', color='C3')

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax

    def scale_location_plot(self, ax=None):
        """
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5);
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False, ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(
                i,
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                color='C3')
        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
        return ax

    def leverage_plot(self, ax=None):
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit
        aka are influential.
        Good to have none outside the curves.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.leverage,
            self.residual_norm,
            alpha=0.5);

        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(self.leverage[i], self.residual_norm[i]),
                color='C3')

        xtemp, ytemp = self.__cooks_dist_line(0.5)  # 0.5 line
        ax.plot(xtemp, ytemp, label="Cook's distance", lw=1, ls='--', color='red')
        xtemp, ytemp = self.__cooks_dist_line(1)  # 1 line
        ax.plot(xtemp, ytemp, lw=1, ls='--', color='red')

        ax.set_xlim(0, max(self.leverage) + 0.01)
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.legend(loc='upper right')
        return ax

    def vif_table(self):
        """
        VIF table

        VIF, the variance inflation factor, is a measure of multicollinearity.
        VIF > 5 for a variable indicates that it is highly collinear with the
        other input variables.
        """
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [variance_inflation_factor(self.xvar, i) for i in range(self.xvar.shape[1])]

        print(vif_df
              .sort_values("VIF Factor")
              .round(2))

    def __cooks_dist_line(self, factor):
        """
        Helper function for plotting Cook's distance curves
        """
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y


def main():
    model1 = general_gravity_model()
    # model2 = partial_gravity_model()
    model3 = no_COVID_model()
    stargazer1 = Stargazer([model1, model3])
    # stargazer2 = Stargazer(model2)
    print(stargazer1.render_latex())  # , '\n', stargazer2.render_latex())
    return


def get_info():
    df = pd.read_csv('Test_file.csv')
    for row in df.itertuples():
        if row.Origin == row.Destination:
            df = df.drop(labels=row.Index, axis=0)
    del df['Origin']
    del df['Destination']
    df['Distance'] = df['Distance'].str.split().str.get(-1)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    df['Distance'] = df['Distance'].astype(float)
    df = dropper(df)
    df = df.dropna()
    # df.to_csv('Test3_file.csv')
    return df


def dropper(df):
    nan_value = float("NaN")
    df.replace(0.0, nan_value, inplace=True)
    df.replace('.', nan_value, inplace=True)
    return df


def general_gravity_model():
    print("General Gravity Model 2014-2020")
    df = get_info()
    df = df.sort_values(by=['Distance'])
    df_endo = df['Commuters']
    del df['Commuters']
    del df['Year']
    log_endo = np.log(df_endo)
    log_exo = np.log(df)
    results = sm.OLS(log_endo, sm.add_constant(log_exo)).fit(cov_type='HC1')
    results2 = sm.RLM(log_endo, sm.add_constant(log_exo), M=sm.robust.norms.HuberT()).fit()
    print(results.summary(), '\n', results2.summary())
    # fig = plt.figure(figsize=(16, 10))
    # fig = sm.graphics.plot_regress_exog(results, 'Destination Pop', fig=fig)
    # plt.gcf()
    # plt.savefig('/Users/main/Downloads/GeneralErrorGrid.png')
    # plt.close()
    # cls = Linear_Reg_Diagnostic(results)
    # fig, ax = cls()
    # plt.gcf()
    # # plt.savefig('/Users/main/Downloads/GeneralDiagnostics.png')
    # plt.close()
    return results2


def partial_gravity_model():
    df = get_info()
    set_of_results = []
    for i in range(2014, 2021, 1):
        print("Partial Gravity Model of %s \n" % i)
        df1 = df[df['Year'] == i]
        df_endo = df1['Commuters']
        del df1['Commuters']
        del df1['Year']
        log_endo = np.log(df_endo)
        log_exo = np.log(df1)
        results = sm.OLS(log_endo, sm.add_constant(log_exo)).fit(cov_type='HC1')
        set_of_results.append(results)
        print(results.summary())
        fig = plt.figure(figsize=(16, 10))
        fig = sm.graphics.plot_regress_exog(results, 'Distance', fig=fig)
        plt.gcf()
        plt.savefig('/Users/main/Downloads/' + str(i) + 'ErrorGrid.png')
        plt.close()
        # cls = Linear_Reg_Diagnostic(results)
        # fig, ax = cls()
        # plt.gcf()
        # plt.savefig('/Users/main/Downloads/' + str(i) + 'Diagnostics.png')
        # plt.close()
    return set_of_results


def no_COVID_model():
    print("Partial Gravity Model 2014-2019 ")
    df = get_info()
    df = df.reset_index()
    del df['index']
    for row in df.itertuples():
        if row.Year == 2020:
            df['Year'][row.Index] = float("NaN")
    df = df.dropna()
    del df['Year']
    df_endo = df['Commuters']
    del df['Commuters']
    log_endo = np.log(df_endo)
    log_exo = np.log(df)
    results = sm.OLS(log_endo, sm.add_constant(log_exo)).fit()
    print(results.summary())
    # fig = plt.figure(figsize=(16, 10))
    # fig = sm.graphics.plot_regress_exog(results, 'Distance', fig=fig)
    # plt.gcf()
    # plt.savefig('/Users/main/Downloads/COVIDErrorGrid.png')
    # plt.close()
    # plt.clf()
    # cls = Linear_Reg_Diagnostic(results)
    # fig, ax = cls()
    # plt.gcf()
    # # plt.savefig('/Users/main/Downloads/COVIDDiagnostics.png')
    # plt.close()
    return results


main()
