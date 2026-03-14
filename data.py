import pandas as pd
import numpy as np
from itertools import groupby


keep_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 
            'FTHG', 'FTAG', 
            'HS', 'HST', 'HC', 'HF', 'HY', 'HR',
            'AS', 'AST', 'AC', 'AF', 'AY', 'AR']
filename2526 = r"C:\coding\SideProjects\football_projects\LaLiga_Pred\data\laliga2526\SP1.csv"
dfnew = pd.read_csv(filename2526)
df = dfnew[[col for col in keep_cols if col in dfnew.columns]]
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
home = df[['Date', "HomeTeam", "FTR"]].copy()
home.columns = ['Date',"Team", "FTR"]
home['Win'] = (home['FTR'] == 'H').astype(int)
home['Draw'] = (home['FTR'] == 'D').astype(int)
home['Loss'] = (home['FTR'] == 'A').astype(int)
away = df[['Date', 'AwayTeam', 'FTR']].copy()
away.columns = ['Date','Team', 'FTR']
away['Win'] = (away['FTR'] == 'A').astype(int)
away['Draw'] = (away['FTR'] == 'D').astype(int)
away['Loss'] = (away['FTR'] == 'H').astype(int)
teams1 = pd.concat([home, away])
summery = teams1.groupby('Team')[['Win','Draw','Loss']].sum()
table = pd.DataFrame(summery)
table['Games'] = table['Win']+table['Draw']+table['Loss']
table['Points'] = table['Win']* 3 +table['Draw']
shotshome = df[["HomeTeam", 'FTHG', "HS", "HST", "HC", 'FTAG', 'AS', 'AST', 'HF','HY',"HR" ]].copy()
shotshome.columns = ["Team",'TG', "TS",  "TST",  "C", 'AG', 'AS', 'ASoT','F','Y','R']
shotsatt = df[['AwayTeam','FTAG', "AS", "AST", "AC", 'FTHG', 'HS','HST','AF','AY','AR']].copy()
shotsatt.columns = ["Team",'TG', "TS",  "TST",  "C", 'AG', 'AS', 'ASoT','F','Y','R']
list1 = pd.concat([shotshome, shotsatt])
ScoringInfo = list1.groupby('Team')[["TG", "TS",  "TST",  "C", 'AG', 'AS', 'ASoT','F','Y','R']].sum()
ScoringInfo["SoT%"] = ScoringInfo["TST"]/ScoringInfo["TS"]
ScoringInfo['Gsh'] = ScoringInfo['TG']/ScoringInfo['TS']
ScoringInfo['GSoT'] = ScoringInfo['TG']/ScoringInfo["TST"]
table = pd.merge(table, ScoringInfo, on='Team', how='outer')
sorted_matches = teams1.sort_values(['Team', 'Date'])
sorted_matches1 = sorted_matches.groupby('Team')['Win'].apply(lambda x: x.tail(10).mean())
dfi = pd.DataFrame.from_dict(sorted_matches1)
dfi.rename(columns={'Win': 'Win_rate10' }, inplace=True)
table = table.merge(dfi, on='Team')
def longest_streak(x):
    streak_lengths = []
    for key, group in groupby(x):
        if key==1:
            biggest = len(list(group))
            streak_lengths.append(biggest)
            pass
    return max(streak_lengths) if streak_lengths else 0
strk_match = sorted_matches.groupby('Team')['Win'].apply(lambda x: longest_streak(x.tail(10)))
strk_match = pd.DataFrame.from_dict(strk_match)
strk_match.rename(columns={'Team': 'Team', 'Win': 'Wstrk_10'}, inplace=True)
table = table.merge(strk_match, on='Team')
table['proxy_xG'] = table['TST'] *0.3 +(table['TS'] - table['TST']) *0.05
table['proxy_xGA'] = table['ASoT'] *0.3 + (table['AS'] - table['ASoT'])*0.05
table['rank'] = table['Points'].rank(ascending=False)
table['G90'] = table['TG']/table['Games']
table['S90'] = table['TS']/table['Games']
table['SoT90'] = table['TST']/table['Games']
table['C90'] = table['C']/table['Games']
table['AG90'] = table['AG']/table["Games"]
table['AS90'] = table['AS']/table['Games']
table['ASoT90']=table['ASoT']/table['Games']
table['F90'] = table['F']/table['Games']
table['Y90'] = table['Y']/table['Games']
table['R90'] = table['R']/table['Games']
unneeded = ["Win", "Draw", "Loss", "Points", "TG", "TS", "TST", "C", "AG", "AS", "ASoT", "F", "Y", "R", "rank"]
table = table.drop(columns=unneeded)
table['season'] = 25
table.reset_index().to_json('2526LaLiga_data20.json', orient='index',indent=4)