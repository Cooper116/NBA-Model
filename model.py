from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog
from nba_api.stats.endpoints.cumestatsteam import CumeStatsTeam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from nba_api.stats.endpoints.teamdashboardbygeneralsplits import TeamDashboardByGeneralSplits
from nba_api.stats.static import teams
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# getting game logs for 18-19 season
df_1819 = LeagueGameLog(season='2018-19').get_data_frames()[0]
games = df_1819.drop(['SEASON_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                 'OREB', 'DREB', 'VIDEO_AVAILABLE', 'PLUS_MINUS', 'WL'], axis=1)

away = games[games['MATCHUP'].str.contains('@')]
home = games[games['MATCHUP'].str.contains('vs.')]

home = home.rename(columns={'TEAM_ABBREVIATION': 'home_team', 'FG_PCT': 'home_fg', 'FG3_PCT': 'home_fg3', 'FT_PCT': 'home_ft',
                     'REB': 'home_reb', 'AST': 'home_ast', 'STL': 'home_stl', 'BLK': 'home_blk', 'TOV': 'home_tov',
                     'PF': 'home_pf', 'PTS': 'home_pts'})

away = away.rename(columns={'TEAM_ABBREVIATION': 'away_team', 'FG_PCT': 'away_fg', 'FG3_PCT': 'away_fg3', 'FT_PCT': 'away_ft',
                     'REB': 'away_reb', 'AST': 'away_ast', 'STL': 'away_stl', 'BLK': 'away_blk', 'TOV': 'away_tov',
                     'PF': 'away_pf', 'PTS': 'away_pts'})

home = home.drop(['MATCHUP'], axis=1)
away = away.drop('MATCHUP', axis=1)

result_1819 = pd.merge(home, away, on='GAME_ID')

# ---------------------------------------------------------------------------------------------------------------------

# getting logs for 19-20 season
df_1920 = LeagueGameLog(season='2019-20').get_data_frames()[0]
games = df_1920.drop(['SEASON_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                 'OREB', 'DREB', 'VIDEO_AVAILABLE', 'PLUS_MINUS', 'WL'], axis=1)

away = games[games['MATCHUP'].str.contains('@')]
home = games[games['MATCHUP'].str.contains('vs.')]

home = home.rename(columns={'TEAM_ABBREVIATION': 'home_team', 'FG_PCT': 'home_fg', 'FG3_PCT': 'home_fg3', 'FT_PCT': 'home_ft',
                     'REB': 'home_reb', 'AST': 'home_ast', 'STL': 'home_stl', 'BLK': 'home_blk', 'TOV': 'home_tov',
                     'PF': 'home_pf', 'PTS': 'home_pts'})

away = away.rename(columns={'TEAM_ABBREVIATION': 'away_team', 'FG_PCT': 'away_fg', 'FG3_PCT': 'away_fg3', 'FT_PCT': 'away_ft',
                     'REB': 'away_reb', 'AST': 'away_ast', 'STL': 'away_stl', 'BLK': 'away_blk', 'TOV': 'away_tov',
                     'PF': 'away_pf', 'PTS': 'away_pts'})

home = home.drop(['MATCHUP'], axis=1)
away = away.drop('MATCHUP', axis=1)

result_1920 = pd.merge(home, away, on='GAME_ID')

# ---------------------------------------------------------------------------------------------------------------------


# get logs for 20-21 season
df_2021 = LeagueGameLog(season='2020-21').get_data_frames()[0]
games = df_2021.drop(['SEASON_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                 'OREB', 'DREB', 'VIDEO_AVAILABLE', 'PLUS_MINUS', 'WL'], axis=1)

away = games[games['MATCHUP'].str.contains('@')]
home = games[games['MATCHUP'].str.contains('vs.')]

home = home.rename(columns={'TEAM_ABBREVIATION': 'home_team', 'FG_PCT': 'home_fg', 'FG3_PCT': 'home_fg3', 'FT_PCT': 'home_ft',
                     'REB': 'home_reb', 'AST': 'home_ast', 'STL': 'home_stl', 'BLK': 'home_blk', 'TOV': 'home_tov',
                     'PF': 'home_pf', 'PTS': 'home_pts'})

away = away.rename(columns={'TEAM_ABBREVIATION': 'away_team', 'FG_PCT': 'away_fg', 'FG3_PCT': 'away_fg3', 'FT_PCT': 'away_ft',
                     'REB': 'away_reb', 'AST': 'away_ast', 'STL': 'away_stl', 'BLK': 'away_blk', 'TOV': 'away_tov',
                     'PF': 'away_pf', 'PTS': 'away_pts'})

home = home.drop(['MATCHUP'], axis=1)
away = away.drop('MATCHUP', axis=1)

result_2021 = pd.merge(home, away, on='GAME_ID')

# combine to create data set for all 3 years to train model
data = pd.concat([result_1819, result_1920, result_2021], ignore_index=True)
data['margin'] = [data.home_pts[i]-data.away_pts[i] for i in range(len(data))]
data['WL'] = [0  if data.home_pts[i] > data.away_pts[i] else 1 for i in range(len(data))]
#data.info()
# print(data.columns)
# split data into features and result (use margin for result)
features = data[['home_fg', 'home_fg3', 'home_ft', 'home_reb', 'home_ast', 'home_stl', 'home_blk', 'home_tov',
                 'away_fg', 'away_fg3', 'away_ft', 'away_reb', 'away_ast', 'away_stl', 'away_blk', 'away_tov']]

y = data[['margin']]
win_loss = data[['WL']]

# split into train and test sets 80/20 split
x_train, x_test, y_train, y_test = train_test_split(features, y, train_size=0.8, test_size=0.2)

# log regression train test split
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(features, win_loss, train_size=0.8, test_size=0.2)

# use training data to train multiple regression model
mlr = LinearRegression()
model = mlr.fit(x_train, y_train)
# y_predict = mlr.predict(x_test)

# Logistic Regression Model
log = LogisticRegression(max_iter=10000)
model_log = log.fit(x_train_log, y_train_log.values.ravel())
# print(log.score(x_test_log, y_test_log))

# plt.scatter(y_test, y_predict, alpha=.4)
# plt.xlabel('Actual Margin')
# plt.ylabel('Predicted Margin')
# plt.plot(y_test, y_test, 'm')
# plt.show()
team_dict = teams.get_teams()

def get_team_stats(name):

    team_info = [team for team in team_dict if team['full_name'] == name][0]
    name_id = team_info['id']

    stat_line = TeamDashboardByGeneralSplits(team_id=name_id).get_data_frames()[0]
    stat_line = stat_line.drop([ 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'GP_RANK', 'W_RANK', 'L_RANK','W_PCT_RANK',
                                 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK',
                                 'FTM_RANK', 'FTA_RANK','FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
                                 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK',
                                 'PLUS_MINUS_RANK', 'CFID', 'CFPARAMS'], axis=1)

    stat_line['rpg'] = stat_line['REB'] / stat_line['GP']
    stat_line['apg'] = stat_line['AST'] / stat_line['GP']
    stat_line['spg'] = stat_line['STL'] / stat_line['GP']
    stat_line['bpg'] = stat_line['BLK'] / stat_line['GP']
    stat_line['tpg'] = stat_line['TOV'] / stat_line['GP']
    stat_line['merge'] = 1

    stat_line = stat_line.drop(['SEASON_YEAR', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                                'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'GROUP_SET', 'GROUP_VALUE', 'GP'], axis=1)

    return stat_line

def predict_regression(home_team, away_team):
    home = get_team_stats(home_team)
    away = get_team_stats(away_team)
    line = pd.merge(home, away, on='merge')
    line = line.drop('merge', axis=1)
    margin = mlr.predict(line)
    print(str(home_team) + ' vs ' + str(away_team) + ': ' + str(margin))
    return margin

def predict_log(home_team, away_team):
    home = get_team_stats(home_team)
    away = get_team_stats(away_team)
    line = pd.merge(home, away, on='merge')
    line = line.drop('merge', axis=1)
    margin = log.predict_proba(line)
    print(str(home_team) + ' vs ' + str(away_team) + ': ' + str(margin))
    return margin



predict_log('Los Angeles Clippers', 'Memphis Grizzlies')
predict_regression('Milwaukee Bucks', 'Charlotte Hornets')





