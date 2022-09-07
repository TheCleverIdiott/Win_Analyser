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
#shows the data types, ML algos can only work with numeric data so any data we feed into the model must be numneric... so object must be converted


del matches["comp"]
del matches["notes"]
#we delete unnecessary data ofr this

matches["date"] = pd.to_datetime(matches["date"])
# crdeating an column with datetim,e and opverwriting the existing column with date time


matches["target"] = (matches["result"] == "W").astype("int")

matches


#creating predictors to build basic ML model that we can add forther complexity to later

matches["venue_code"] = matches["venue"].astype("category").cat.codes
#converting from string to categories and then converting those categoties to numbers
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches




from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']

predictors = ["venue_code", "opp_code", "hour", "day_code"]


preds = rf.predict(test[predictors])
