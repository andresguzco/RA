import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = get_info()
    descriptive_stats(df)
    return


def descriptive_stats(df):
    df = df.astype(int)
    del df['Origin']
    del df['Destination']
    print(df.describe().to_latex(), '\n')
    print(df.corr().to_latex(), '\n', df.cov().to_latex(), '\n\n')
    # stat_plotter(df, "General")
    for i in range(2014, 2021, 1):
        print("Descriptive Statistics of %s \n" % i)
        df1 = df[df['Year'] == i]
        del df1['Year']
        print(df1.describe().to_latex(), '\n')
        print(df1.corr().to_latex(), '\n', df1.cov().to_latex(), '\n\n')
        stat_plotter(df1, i)
    return


def stat_plotter(df, i):
    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.scatter(df["Distance"], df["Origin Pop"], color='b', alpha=0.20)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Origin Pop')
    plt.gcf()
    plt.savefig('Results/OrDistance' + str(i) + '.png')
    plt.close()
    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.scatter(df["Distance"], df["Destination Pop"], color='b', alpha=0.20)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Destination Pop')
    plt.gcf()
    plt.savefig('Results/DestDistance' + str(i) + '.png')
    plt.close()
    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.scatter(df["Distance"], df["Commuters"], color='b', alpha=0.20)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Commuters')
    plt.gcf()
    plt.savefig('Results/DistCom' + str(i) + '.png')
    plt.close()
    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.scatter(df["Destination Pop"], df["Commuters"], color='b', alpha=0.20)
    ax.set_xlabel('Destination Pop')
    ax.set_ylabel('Commuters')
    plt.gcf()
    plt.savefig('Results/DestCom' + str(i) + '.png')
    plt.close()
    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.scatter(df["Origin Pop"], df["Commuters"], color='b', alpha=0.20)
    ax.set_xlabel('Origin Pop')
    ax.set_ylabel('Commuters')
    plt.gcf()
    plt.savefig('Results/OrCom' + str(i) + '.png')
    plt.close()


def get_info():
    df = pd.read_csv('Databases/Test_file.csv')
    for row in df.itertuples():
        if row.Origin == row.Destination:
            df = df.drop(labels=row.Index, axis=0)
    df['Distance'] = df['Distance'].str.split().str.get(-1)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    df = dropper(df)
    df = df.dropna()
    return df


def dropper(df):
    nan_value = float("NaN")
    df.replace(0.0, nan_value, inplace=True)
    return df


main()
