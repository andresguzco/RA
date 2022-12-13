import pandas as pd
import numpy as np


def main():

    return


def dropper(df):
    nan_value = float("NaN")
    df.replace(0.0, nan_value, inplace=True)
    return df


def formatter(input_df):
    df = input_df
    del df['ID']
    df.columns = ['Origin', 'Destination', 'Year', 'Commuters', 'Distance']
    df['Origin'] = df['Origin'].map(lambda x: x.lstrip('GM'))
    df['Destination'] = df['Destination'].map(lambda x: x.lstrip('GM'))
    df = dropper(df)
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
    df['Origin'] = df['Origin'].astype(int)
    df['Destination'] = df['Destination'].astype(int)
    df['Distance'] = df['Distance'].str.split().str.get(-1)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    df['Distance'] = df['Distance'].astype(float)
    return df

def commuting_matrix_creator(input_df, year):
    municipalities = input_df['Origin'].unique().tolist()
    output_df = pd.DataFrame(columns=municipalities, index=municipalities)
    for i in input_df.itertuples():
        output_df.at[i.Origin, i.Destination] = i.Commuters
    output_df = output_df.fillna(0)
    output_df = output_df.sort_index()
    output_df = output_df.sort_index(axis=1)
    return output_df


def population_formatter(pathway):
    pop_df = pd.read_csv(f'{pathway}', sep=';')
    del pop_df['ID']
    del pop_df['Sex']
    pop_df.columns = ['Region', 'Year', 'Population']
    pop_df.dropna(subset=["Population"], inplace=True)
    pop_df['Region'] = pop_df['Region'].map(lambda x: x.lstrip('GM'))
    pop_df['Region'] = pop_df['Region'].astype(int)
    pop_df = pop_df.reset_index()
    del pop_df['index']
    for row in pop_df.iterrows():
        X = row[1]['Year']
        X = X.split('JJ')[0]
        pop_df.at[row[0], 'Year'] = int(X)
    return pop_df


def format_for_regression(comm_df, pop_df, year):
    df = pre_formatter(comm_df.copy())
    df2 = pop_df.loc[pop_df['Year'] == int(year)]
    for i in df.itertuples():
        for j in df2.itertuples():
            if i.Origin == j.Region:
                df.at[i.Index, 'Origin Pop'] = np.log(j.Population)
                if i.Destination == j.Region:
                    df.at[i.Index, 'Destination Pop'] = np.log(j.Population)
            elif i.Destination == j.Region:
                df.at[i.Index, 'Destination Pop'] = np.log(j.Population)
    pathway = 'Databases/Regression_DataFrame_' + str(year) + '.csv'
    df.to_csv(f'{pathway}')
    return df


def pre_formatter(input_df):
    df = input_df.copy()
    for i in df.itertuples():
        if i.Distance != 0:
            df.at[i.Index, 'Distance'] = np.log(i.Distance)
        if i.Commuters != 0:
            df.at[i.Index, 'Commuters'] = np.log(i.Commuters)
    return df


def get_info(pathway, year, pop_df):
    rawdf = formatter(pd.read_csv(f'{pathway}', sep=";"))
    df = commuting_matrix_creator(rawdf, year)
    df_reg = format_for_regression(rawdf, pop_df, year)
    return df, df_reg


pop_general = population_formatter('Databases/Population 2014-2020.csv')


com_matrix_14, df_14_reg = get_info('Databases/Mobility 2014.csv', 2014, pop_general)
com_matrix_15, df_15_reg = get_info('Databases/Mobility 2015.csv', 2015, pop_general)
com_matrix_16, df_16_reg = get_info('Databases/Mobility 2016.csv', 2016, pop_general)
com_matrix_17, df_17_reg = get_info('Databases/Mobility 2017.csv', 2017, pop_general)
com_matrix_18, df_18_reg = get_info('Databases/Mobility 2018.csv', 2018, pop_general)
com_matrix_19, df_19_reg = get_info('Databases/Mobility 2019.csv', 2019, pop_general)
com_matrix_20, df_20_reg = get_info('Databases/Mobility 2020.csv', 2020, pop_general)


aggregate_df_reg = pd.concat([df_14_reg, df_15_reg, df_16_reg, df_17_reg, df_18_reg, df_19_reg, df_20_reg])
aggregate_df_reg.to_csv('Databases/Test_file.csv', index=False)


## Changes in borders between municipalities

# Some years, the Dutch government change municipality's borders by merging small ones into bigger ones for a myriad of reasons. We will identify these year to year modifications in the matrices.

municipalities_14 = com_matrix_14.columns.values.tolist()
municipalities_15 = com_matrix_15.columns.values.tolist()
municipalities_16 = com_matrix_16.columns.values.tolist()
municipalities_17 = com_matrix_17.columns.values.tolist()
municipalities_18 = com_matrix_18.columns.values.tolist()
municipalities_19 = com_matrix_19.columns.values.tolist()
municipalities_20 = com_matrix_20.columns.values.tolist()


dif_14_15 = list(set(municipalities_14).symmetric_difference(set(municipalities_15)))
print(dif_14_15,'\n', 'The modification are:',len(dif_14_15), '\n',
      'The actual difference should be:',len(com_matrix_14) - len(com_matrix_15))


dif_15_16 = list(set(municipalities_15).symmetric_difference(set(municipalities_16)))
print(dif_15_16,'\n', 'The modification are:', len(dif_15_16), '\n',
      'The actual difference should be:', len(com_matrix_15) - len(com_matrix_16))


dif_16_17 = list(set(municipalities_16).symmetric_difference(set(municipalities_17)))
print(dif_16_17,'\n', 'The modification are:', len(dif_16_17), '\n',
      'The actual difference should be:', len(com_matrix_16) - len(com_matrix_17))


dif_17_18 = list(set(municipalities_17).symmetric_difference(set(municipalities_18)))
print(dif_17_18,'\n', 'The modification are:', len(dif_17_18), '\n',
      'The actual difference should be:', len(com_matrix_17) - len(com_matrix_18))


dif_18_19 = list(set(municipalities_18).symmetric_difference(set(municipalities_19)))
print(dif_18_19,'\n', 'The modification are:', len(dif_18_19), '\n',
      'The actual difference should be:', len(com_matrix_18) - len(com_matrix_19))


dif_19_20 = list(set(municipalities_19).symmetric_difference(set(municipalities_20)))
print(dif_19_20,'\n', 'The modification are:', len(dif_19_20), '\n',
      'The actual difference should be:', len(com_matrix_19) - len(com_matrix_20))


## Backcasting border changes

# Every year, some municipalities disappear. Sometimes they merge with other existent municipalities or sometimes a new bigger municipality is created by the fusion of others. In this section we will write a small funciton that will take a given year commuting matrix, the disappearing municipalities and the new one. Then we will create another function that does the same for the other type of DataFrame, the one that also has the distance between municipalities.

def commuting_matrix_updater(commuting_matrix, disappearing_municipalities, new_municipalities):
    df = commuting_matrix.copy()
    municipalities = df.columns.values.tolist()
    for i in range(len(new_municipalities)):
        if new_municipalities[i] in municipalities:
            for j in range(len(disappearing_municipalities[i])):
                df[new_municipalities[i]] += df[disappearing_municipalities[i][j]]
                del df[disappearing_municipalities[i][j]]
                df = df.T
                df[new_municipalities[i]] += df[disappearing_municipalities[i][j]]
                del df[disappearing_municipalities[i][j]]
        else:
            for j in range(len(disappearing_municipalities[i])):
                df[new_municipalities[i]] = df[disappearing_municipalities[i][j]]
                del df[disappearing_municipalities[i][j]]
                df = df.T
                df[new_municipalities[i]] = df[disappearing_municipalities[i][j]]
                del df[disappearing_municipalities[i][j]]
    return df

# def regression_dataframe_updater(regression_dataframe, disappearing_municipalities, new_municipalities):
#     df = regression_dataframe.copy()
#     for i in df.itertuples():
#         if i.Index ==
#     return


# Here, we already have a function that can modify the commuting matrix in the way we mentioned. To avoid problems
# with the division of municipalities and merging of many into one, we can write a separate loop.

# 2014 - 2020

disappearing_municipalities = [[491, 608, 623, 643, 644], [265, 282], [365, 458], [568, 612]]
target_municipality = [1931, 241, 361, 1930]
com_matrix_14 = commuting_matrix_updater(com_matrix_14, disappearing_municipalities, target_municipality)

del com_matrix_14[1671]
com_matrix_14 = com_matrix_14.T
del com_matrix_14[1671]

disappearing_municipalities = [[381, 424, 425], [478], [241], [1921]]
target_municipality = [1942, 385, 1945, 1940]
com_matrix_14 = commuting_matrix_updater(com_matrix_14, disappearing_municipalities, target_municipality)

disappearing_municipalities = [[844, 846, 860]]
target_municipality = [1948]
com_matrix_14 = commuting_matrix_updater(com_matrix_14, disappearing_municipalities, target_municipality)

disappearing_municipalities = [[7, 48], [1987, 18, 40], [81, 140], [196], [63, 70, 1908]]
target_municipality = [1950, 1952, 80, 299, 1949]
com_matrix_14 = commuting_matrix_updater(com_matrix_14, disappearing_municipalities, target_municipality)

disappearing_municipalities = [[53, 5, 1651, 1663], [545, 707, 620], [15, 22, 25, 56], [58, 79, 1722], [689, 1927],
                               [236, 304, 733], [17, 9], [393], [576], [584, 585, 588, 611, 617], [738, 870, 874],
                               [962, 881, 951]]
target_municipality = [1966, 1961, 1969, 1970, 1978, 1960, 14, 394, 575, 1963, 1959, 1954]
com_matrix_14 = commuting_matrix_updater(com_matrix_14, disappearing_municipalities, target_municipality)

pathway = 'Databases/Commuting_Matrix_2014.csv'
com_matrix_14.to_csv(f'{pathway}')

print(len(com_matrix_14))

# 2015 - 2020

disappearing_municipalities = [[381, 424, 425], [478], [241], [1921]]
target_municipality = [1942, 385, 1945, 1940]
com_matrix_15 = commuting_matrix_updater(com_matrix_15, disappearing_municipalities, target_municipality)

disappearing_municipalities = [[844, 846, 860]]
target_municipality = [1948]
com_matrix_15 = commuting_matrix_updater(com_matrix_15, disappearing_municipalities, target_municipality)

disappearing_municipalities = [[7, 48], [1987, 18, 40], [81, 140], [196], [63, 70, 1908]]
target_municipality = [1950, 1952, 80, 299, 1949]
com_matrix_15 = commuting_matrix_updater(com_matrix_15, disappearing_municipalities, target_municipality)

disappearing_municipalities = [[53, 5, 1651, 1663], [545, 707, 620], [15, 22, 25, 56], [58, 79, 1722], [689, 1927],
                               [236, 304, 733], [17, 9], [393], [576], [584, 585, 588, 611, 617], [738, 870, 874],
                               [962, 881, 951]]
target_municipality = [1966, 1961, 1969, 1970, 1978, 1960, 14, 394, 575, 1963, 1959, 1954]
com_matrix_15 = commuting_matrix_updater(com_matrix_15, disappearing_municipalities, target_municipality)

pathway = 'Databases/Commuting_Matrix_2015.csv'
com_matrix_15.to_csv(f'{pathway}')

print(len(com_matrix_15))

# 2016 - 2020

disappearing_municipalities = [[844, 846, 860]]
target_municipality = [1948]
com_matrix_16 = commuting_matrix_updater(com_matrix_16, disappearing_municipalities, target_municipality)

disappearing_municipalities = [[7, 48], [1987, 18, 40], [81, 140], [196], [63, 70, 1908]]
target_municipality = [1950, 1952, 80, 299, 1949]
com_matrix_16 = commuting_matrix_updater(com_matrix_16, disappearing_municipalities, target_municipality)

disappearing_municipalities = [[53, 5, 1651, 1663], [545, 707, 620], [15, 22, 25, 56], [58, 79, 1722], [689, 1927],
                               [236, 304, 733], [17, 9], [393], [576], [584, 585, 588, 611, 617], [738, 870, 874],
                               [962, 881, 951]]
target_municipality = [1966, 1961, 1969, 1970, 1978, 1960, 14, 394, 575, 1963, 1959, 1954]
com_matrix_16 = commuting_matrix_updater(com_matrix_16, disappearing_municipalities, target_municipality)

pathway = 'Databases/Commuting_Matrix_2016.csv'
com_matrix_16.to_csv(f'{pathway}')

print(len(com_matrix_16))

# 2017 - 2020

disappearing_municipalities = [[7, 48], [1987, 18, 40], [81, 140], [196], [63, 70, 1908]]
target_municipality = [1950, 1952, 80, 299, 1949]
com_matrix_17 = commuting_matrix_updater(com_matrix_17, disappearing_municipalities, target_municipality)

disappearing_municipalities = [[53, 5, 1651, 1663], [545, 707, 620], [15, 22, 25, 56], [58, 79, 1722], [689, 1927],
                               [236, 304, 733], [17, 9], [393], [576], [584, 585, 588, 611, 617], [738, 870, 874],
                               [962, 881, 951]]
target_municipality = [1966, 1961, 1969, 1970, 1978, 1960, 14, 394, 575, 1963, 1959, 1954]
com_matrix_17 = commuting_matrix_updater(com_matrix_17, disappearing_municipalities, target_municipality)

pathway = 'Databases/Commuting_Matrix_2017.csv'
com_matrix_17.to_csv(f'{pathway}')

print(len(com_matrix_17))

# 2018 - 2020

disappearing_municipalities = [[53, 5, 1651, 1663], [545, 707, 620], [15, 22, 25, 56], [58, 79, 1722], [689, 1927],
                               [236, 304, 733], [17, 9], [393], [576], [584, 585, 588, 611, 617], [738, 870, 874],
                               [962, 881, 951]]
target_municipality = [1966, 1961, 1969, 1970, 1978, 1960, 14, 394, 575, 1963, 1959, 1954]
com_matrix_18 = commuting_matrix_updater(com_matrix_18, disappearing_municipalities, target_municipality)

pathway = 'Databases/Commuting_Matrix_2018.csv'
com_matrix_18.to_csv(f'{pathway}')

print(len(com_matrix_18))

# 2019 - 2020

pathway = 'Databases/Commuting_Matrix_2019.csv'
com_matrix_19.to_csv(f'{pathway}')

# 2020

pathway = 'Databases/Commuting_Matrix_2020.csv'
com_matrix_20.to_csv(f'{pathway}')


# There's no changes in municipalities between 2019 and 2020, therefore the DataFrames stay unmodified.

if __name__ == "__main__":
    main()