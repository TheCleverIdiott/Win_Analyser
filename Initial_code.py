import pandas as pd

matches = pd.read_csv("matches.csv", index_col=0)  #Download and save the data from IBM Watson
matches.head()

shape = matches.shape


# lets assume EPL based.... so 38 matches each season and 20 teams; Given IBM data has data for 2 seasons
# so we must have 38*20*2 = 1520 match datas
# if not we have to investigate missing data


matches["team"].value_counts()
# gives each squad and how many matches they have played
# 3 teams delegated and 3 pulled up
# expected 6 teams ot have fewer matches


matches[matches["team"] == "Liverpool"].sort_values("date")
# we see that we have 7 teams with less matches
# we investigate the seventh team (here liverpool)
# just selecting teams where the rows are liverpool


matches["round"].value_counts()
# the round column tells you which match week it was played
#after running this we can see how many matches were played were in each week
# we should have 38 each match week



matches.dtypes
#shows the data types, ML algos can only work with numeric data so any data we feed into the model must be numeric... so object must be converted


del matches["comp"]
del matches["notes"]
#we delete unnecessary data ofr this

matches["date"] = pd.to_datetime(matches["date"])
# crdeating an column with datetime and operwriting the existing column with date time


matches["target"] = (matches["result"] == "W").astype("int")

matches


#creating predictors to build basic ML model that we can add further complexity to later

matches["venue_code"] = matches["venue"].astype("category").cat.codes
#converting venue column from string to categorical data type and then converting those categories to integers 0 and 1

matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches




from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
#series of decision trees where each decision tree has slightly different parameters.
# n_estimators=50 is the number of decision trees we want to train.
# n is directly proportional to run-time and accuracy.
# min_samples_split=10 is the number of samples we wann ahave in a leaf of the tree before we split the node.
# ther higher min_samples_split is the less likely we are to overfit but the lower our accuracy on the training data.
# random state means thatif we run the run tyhe random forest multiple times we will get the same data as long as the data remains the same.


train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
#splitting our training and test data, whiuch is date-time data; train is using anything tyhat came before 2022 (past) and test is anything in 2022(present); then comopare the values to get accuarcy.
# so basically we are using the past to predict the present and tallying it with the presently known values to get how accuarte our result is. This will tell us how accurate our model will be if we use present data to predict the actual future.


predictors = ["venue_code", "opp_code", "hour", "day_code"]
#passing predictor parameters.

rf.fit(train[predictors], train["target"])
# the .fit methid is basically going to train the random forest model with the passed parameters to predict the target, which is 0 if the team lost or drew and 1 if they won.

preds = rf.predict(test[predictors])
#generating predictions using the .predict method by passing our test data.


# Now we need to figure out a way to determine the accuaracy of the model.
# Important choice, so we'rte gonna try a couple of different metrics and see which one makews more sense.
from sklearn.metrics import accuracy_score
precision_score(test["target"], preds)

grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester City").sort_values("date")

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group
  
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling

matches_rolling = matches_rolling.droplevel('team')
matches_rolling

matches_rolling.index = range(matches_rolling.shape[0])
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error
combined, error = make_predictions(matches_rolling, predictors + new_cols)

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
combined.head(10)

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)
combined["new_team"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
merged

merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()









