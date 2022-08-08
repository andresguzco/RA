import pandas as pd


def main():
    df1 = get_info1()
    df2 = get_info2()
    # print(df1, df2)
    df = format_for_regression(df1, df2)
    print(df)
    df.to_csv('Test_file.csv', index=False)
    return


def dropper(df):
    nan_value = float("NaN")
    df.replace(0.0, nan_value, inplace=True)
    return df


def get_info1():
    df1 = pd.read_csv('Databases/Mobility 2014.csv', sep=";")
    df2 = pd.read_csv('Databases/Mobility 2015-2016.csv', sep=";")
    df3 = pd.read_csv('Databases/Mobility 2017-2018.csv', sep=";")
    df4 = pd.read_csv('Databases/Mobility 2019-2020.csv', sep=";")
    df = [df1, df2, df3, df4]
    final_df = pd.concat(df)
    del final_df['ID']
    final_df.columns = ['Origin', 'Destination', 'Year', 'Commuters', 'Distance']
    final_df['Origin'] = final_df['Origin'].map(lambda x: x.lstrip('GM'))
    final_df['Destination'] = final_df['Destination'].map(lambda x: x.lstrip('GM'))
    df = dropper(final_df)
    df.dropna(subset=["Commuters"], inplace=True)
    df = df.reset_index()
    del df['index']
    for row in df.iterrows():
        X = row[1]['Year']
        X = X.split('MM')[0]
        Y = row[1]['Commuters']
        Y = Y * 1000
        df.at[row[0], 'Year'] = X
        df.at[row[0], 'Commuters'] = Y
    return df


def get_info2():
    df = pd.read_csv('Databases/Population 2014-2020.csv', sep=';')
    del df['ID']
    del df['Sex']
    df.columns = ['Region', 'Year', 'Population']
    df.dropna(subset=["Population"], inplace=True)
    df['Region'] = df['Region'].map(lambda x: x.lstrip('GM'))
    df = df.reset_index()
    del df['index']
    for row in df.iterrows():
        X = row[1]['Year']
        X = X.split('JJ')[0]
        df.at[row[0], 'Year'] = X
    return df


def format_for_regression(df1, df2):
    df = df1
    df['Origin Pop'] = ''
    df['Destination Pop'] = ''
    for row in df.itertuples():
        for index in df2.itertuples():
            if row.Origin == index.Region and row.Year == index.Year:
                df.at[row.Index, 'Origin Pop'] = index.Population
                if row.Destination == index.Region:
                    df.at[row.Index, 'Destination Pop'] = index.Population
            elif row.Destination == index.Region and row.Year == index.Year:
                df.at[row.Index, 'Destination Pop'] = index.Population
    return df


main()


neg_data <- data.frame(transport, lambda, rel, welfare,
                       w_man_h, w_man_f, w_farm_h, w_farm_f)



top_line <- neg_data[neg_data$transport == "1.5", ]
bottom_line <- neg_data[neg_data$transport == "2", ]
mid_line <- neg_data[neg_data$transport == "1.75", ]



#First Plot
ggplot(neg_data) + aes(lambda, rel, group = transport) + geom_line(size = 0.5, colour="grey", alpha = 0.5) +
geom_line(data = top_line, aes(x = lambda, y = rel, group = transport, colour = "steelblue"), size = 1) +
geom_line(data = bottom_line, aes(x = lambda, y = rel, group = transport, colour = "black"), size = 1) +
geom_line(data = mid_line, aes(x = lambda, y = rel, group = transport, colour = "red"), size = 1) +
scale_colour_discrete(name = "Transportation costs", labels = c("High", "Medium", "Low")) +
geom_hline(yintercept = 1, size = 1, colour = "red", linetype = 4) +
theme_economist() +
labs(title ="Wiggle diagram", y = "Relative real wage",
     subtitle = "Changes in relative real wage with varying lambda and transportation costs")



#Second Plot
ggplot(equilibria) + aes(t_vec, lam_vec) +
geom_point(aes(colour = factor(stable))) +
theme_economist() +
theme(legend.title=element_blank()) +
scale_colour_discrete(breaks = c("0", "1"), labels=c("Unstable equilibrium", "Stable equilibrium")) +
labs(title ="Tomahawk", y = "lambda", x = "transportation costs")
