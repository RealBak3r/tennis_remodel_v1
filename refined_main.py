import random
random.seed(1)
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



data = pd.read_csv('/Users/user/Desktop/tennis_remodel_v1/data/atp_matches_2018.csv')


#print(data.shape)
#print(data.dtypes)
#print(data.isnull().sum())
#print(data.value_counts())
#print(data.describe())

#cleaning the data step 1
data[['winner_hand', 'loser_hand']] = (data[['winner_hand', 'loser_hand']] == 'L').astype(bool)

#print(data['winner_hand'].unique())

#print(data['winner_hand'].isnull().sum())
def randomizer(row):
    flip = random.choice([False, True])
    if flip:
        row['player_1_rank'] = row['winner_rank']
        row['player_2_rank'] = row['loser_rank']
        row['player_1_hand'] = row['winner_hand']
        row['player_2_hand'] = row['loser_hand']
        row['player_1_ioc'] = row['winner_ioc']
        row['player_2_ioc'] = row['loser_ioc']
        row['player_1_ht'] = row['winner_ht']
        row['player_2_ht'] = row['loser_ht']
        row['player_1_age'] = row['winner_age']
        row['player_2_age'] = row['loser_age']
        row['player_1_rank'] = row['winner_rank']
        row['player_2_rank'] = row['loser_rank']
        row['player_1_points'] = row['winner_rank_points']
        row['player_2_points'] = row['loser_rank_points']
        row['player_1_won'] = 1
    else:
        row['player_1_rank'] = row['loser_rank']
        row['player_2_rank'] = row['winner_rank']
        row['player_1_hand'] = row['loser_hand']
        row['player_2_hand'] = row['winner_hand']
        row['player_1_ioc'] = row['loser_ioc']
        row['player_2_ioc'] = row['winner_ioc']
        row['player_1_ht'] = row['loser_ht']
        row['player_2_ht'] = row['winner_ht']
        row['player_1_age'] = row['loser_age']
        row['player_2_age'] = row['winner_age']
        row['player_1_rank'] = row['loser_rank']
        row['player_2_rank'] = row['winner_rank']
        row['player_1_points'] = row['loser_rank_points']
        row['player_2_points'] = row['winner_rank_points']
        row['player_1_won'] = 0
    return row

def combiner(row):
    row['points_difference'] = row['num__player_1_points'] - row['num__player_2_points']
    row['age_difference'] = row['num__player_1_age'] - row['num__player_2_age']
    row['ht_difference'] = row['num__player_1_ht'] - row['num__player_2_ht']
    row['rank_difference'] = row['num__player_1_rank'] - row['num__player_2_rank']
    return row

#print(data['round'].unique())

#print(data.columns.tolist())

#apply the randomizer
data = data.apply(randomizer, axis='columns')

#print(data.isnull().sum())

#computing the missing values step 2

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), ['player_1_rank','player_2_rank', 'player_1_ht', 'player_2_ht', 'player_1_age', 'player_2_age', 'player_1_points', 'player_2_points']),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['surface', 'draw_size', 'tourney_name', 'tourney_level', 'best_of', 'round']),
    ('pass', 'passthrough', ['player_1_won'])
])
#at this point I have a found a VERY SMALL leakage as the hand column seems to be clean now as it used to not be that way. it may be because of my step 1 line 1.
#FIX: I changed the == 'R' to 'L' as if it was ever null which would be a False as null is auto False, it would turn into the most frequent because 83% of tennis players are right handed so I decided to make right False and left Right.

data_processed = pd.DataFrame(preprocessor.fit_transform(data), columns=preprocessor.get_feature_names_out())
data_processed = data_processed.apply(combiner, axis='columns')

incomplete_pipeline = Pipeline(steps=[
    ('model', XGBClassifier(random_state=1))
])

chosen = ['rank_difference','points_difference', 'ht_difference', 'age_difference','num__player_1_rank', 'num__player_2_rank', 'num__player_1_ht', 'num__player_2_ht', 'num__player_1_age', 'num__player_2_age', 'num__player_1_points', 'num__player_2_points']
#,'num__player_1_rank', 'num__player_2_rank', 'num__player_1_ht', 'num__player_2_ht', 'num__player_1_age', 'num__player_2_age', 'num__player_1_points', 'num__player_2_points',

y = data['player_1_won']

x = data_processed[chosen]

X_train, X_val, y_train, y_val = train_test_split(x,y, test_size=0.20, random_state=1)



param_grid = {
    'model__n_estimators': [30,40,45,55,65,75],
    'model__max_depth': [2,3,4,5],
    'model__learning_rate': [0.1,0.2,0.3]
}


grid_search = GridSearchCV(incomplete_pipeline, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)
prediction = grid_search.predict_proba(X_val)
print(prediction)
#print(accuracy_score(y_val, prediction))
#print(grid_search.best_score_)
#print(grid_search.best_params_)


#this is the score with all of the columns available so now we have to get better than that and start removing noise
#0.5950704225352113
#0.6228295193169856
#{'model__learning_rate': 0.1, 'model__max_depth': 5, 'model__n_estimators': 55}


#this is the result with only the differences
#0.6161971830985915
#0.6192663054071966
#{'model__learning_rate': 0.1, 'model__max_depth': 2, 'model__n_estimators': 55}

#this is the result with : ['rank_difference','points_difference', 'ht_difference', 'age_difference','num__player_1_rank', 'num__player_2_rank', 'num__player_1_ht', 'num__player_2_ht', 'num__player_1_age', 'num__player_2_age', 'num__player_1_points', 'num__player_2_points']
#0.6373239436619719
#0.6316517874546801
#{'model__learning_rate': 0.1, 'model__max_depth': 2, 'model__n_estimators': 55}

#step 3 removing noise and get a better accuracy
#print(data_processed['points_difference'])



#THIS PART HAS NOT BEEN DONE BY ME
"""model = grid_search.best_estimator_.named_steps['model']

importance = model.feature_importances_

result_import = pd.Series(importance, index=chosen).sort_values(ascending=False)

print(result_import)

result_perm = permutation_importance(grid_search, X_val, y_val, n_repeats=10, random_state=1)
pd.Series(result_perm.importances_mean, index=chosen).sort_values(ascending=False)

data_result = pd.DataFrame({
    'mean': result_perm.importances_mean,
    'std': result_perm.importances_std
        }, index=chosen).sort_values('mean', ascending=False)



print(data_result)"""


#corr = data_processed[chosen].corr()['pass__player_1_won'].sort_values(ascending=False)
#print(corr)

#print(data_processed[chosen].sum().sort_values(ascending=False).astype(int))

