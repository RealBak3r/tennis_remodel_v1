import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import random

#sklearn imports

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance





data = pd.read_csv('/Users/user/Desktop/tennis_remodel_v1/data/atp_matches_2018.csv')

def randoming(row):
    flip = random.choice([True, False])
    if flip:
        row['player_1_hand'] = row['winner_hand']
        row['player_2_hand'] = row['loser_hand']
        row['player_1_ht'] = row['winner_ht']
        row['player_2_ht'] = row['loser_ht']
        row['player_1_age'] = row['winner_age']
        row['player_2_age'] = row['loser_age']
        row['player_1_rank'] = row['winner_rank']
        row['player_2_rank'] = row['loser_rank']
        row['player_1_points'] = row['winner_rank_points']
        row['player_2_points'] = row['loser_rank_points']
        row['player_1_ioc'] = row['winner_ioc']
        row['player_2_ioc'] = row['loser_ioc']
        row['player_1_won'] = 1

    else:
        row['player_1_hand'] = row['loser_hand']
        row['player_2_hand'] = row['winner_hand']
        row['player_1_ht'] = row['loser_ht']
        row['player_2_ht'] = row['winner_ht']
        row['player_1_age'] = row['loser_age']
        row['player_2_age'] = row['winner_age']
        row['player_1_rank'] = row['loser_rank']
        row['player_2_rank'] = row['winner_rank']
        row['player_1_points'] = row['loser_rank_points']
        row['player_2_points'] = row['winner_rank_points']
        row['player_1_ioc'] = row['loser_ioc']
        row['player_2_ioc'] = row['winner_ioc']
        row['player_1_won'] = 0

    return row


data = data.apply(randoming, axis='columns')

data[['player_1_age','player_2_age']] = data[['player_1_age','player_2_age']].round(2)
data[['player_1_hand', 'player_2_hand']] = (data[['player_1_hand', 'player_2_hand']] == 'R').astype(float)

preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'),['player_1_ht','player_2_ht', 'player_1_age', 'player_2_age', 'player_1_rank', 'player_2_rank', 'player_1_points', 'player_2_points']),
    ('fre', SimpleImputer(strategy='most_frequent'), ['player_1_hand', 'player_2_hand']),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),['surface', 'tourney_level', 'tourney_name', 'player_1_ioc', 'player_2_ioc', 'draw_size', 'best_of']),
    ('pass', 'passthrough', ['player_1_won'])
])

data_processed = pd.DataFrame(preprocessor.fit_transform(data), columns=preprocessor.get_feature_names_out())

#print(data.isnull().sum())

#print(data_processed.columns.tolist())

#print(data['best_of'].unique())

full_pipeline = Pipeline(steps=[
    ('model', XGBClassifier(random_state=1))
])


print(data_processed.columns.tolist())
chosen04 = ["num__player_1_points", "num__player_2_rank", "fre__player_1_hand", "cat__tourney_name_New York", "cat__player_2_ioc_JPN", "cat__tourney_name_Davis Cup WG R1: HUN vs BEL", "cat__player_2_ioc_MAR", "cat__player_1_ioc_SUI", "cat__tourney_name_Pune", "cat__tourney_name_Quito", "cat__player_1_ioc_MEX", "cat__tourney_name_Davis Cup G2 R1: TUR vs ZIM", "cat__player_2_ioc_TUR", "cat__tourney_name_Davis Cup G2 R2: MEX vs PER", "cat__tourney_name_Davis Cup G2 R1: EGY vs NOR", "cat__draw_size_32", "cat__player_2_ioc_UKR", "cat__player_2_ioc_DOM", "cat__player_2_ioc_MDA", "cat__tourney_name_Barcelona", "cat__player_2_ioc_TUN", "cat__tourney_name_Marrakech", "cat__player_2_ioc_CRO", "fre__player_2_hand", "cat__tourney_name_Davis Cup G1 R1: COL vs BAR", "cat__tourney_name_Davis Cup WG R1: AUS vs GER", "cat__tourney_name_Davis Cup G1 R1: CHN vs NZL", "cat__tourney_name_Davis Cup G1 R2: AUT vs RUS", "cat__tourney_name_Davis Cup G1 R2: CZE vs ISR", "cat__player_1_ioc_PUR", "cat__tourney_name_Davis Cup WG QF: USA vs BEL", "cat__player_2_ioc_RSA", "cat__player_2_ioc_AUT", "cat__tourney_name_Davis Cup G2 R1: PHI vs INA", "cat__tourney_name_Davis Cup G1 R1: AUT vs BLR", "cat__player_2_ioc_ESP", "cat__player_1_ioc_LUX", "cat__player_1_ioc_CZE", "cat__draw_size_128", "cat__player_1_ioc_BAR", "num__player_2_ht", "num__player_2_points"]
chosen05 = ["pass__player_1_won", "num__player_1_points", "num__player_2_rank", "fre__player_1_hand", "cat__tourney_name_New York", "cat__player_2_ioc_JPN", "cat__tourney_name_Davis Cup WG R1: HUN vs BEL", "cat__player_2_ioc_MAR", "cat__player_1_ioc_SUI", "cat__player_2_ioc_AUT", "cat__tourney_name_Davis Cup G2 R1: PHI vs INA", "cat__tourney_name_Davis Cup G1 R1: AUT vs BLR", "cat__player_2_ioc_ESP", "cat__player_1_ioc_LUX", "cat__player_1_ioc_CZE", "cat__draw_size_128", "cat__player_1_ioc_BAR", "num__player_2_ht", "num__player_2_points"]
chosen06 = ["pass__player_1_won", "num__player_1_points", "num__player_2_rank", "fre__player_1_hand", "cat__tourney_name_New York", "num__player_2_ht", "num__player_2_points"]
chosen_50 = [
    "num__player_1_points",
    "num__player_2_rank",
    "fre__player_1_hand",
    "cat__draw_size_32",
    "fre__player_2_hand",
    "cat__player_2_ioc_ESP",
    "cat__draw_size_128",
    "num__player_2_ht",
    "num__player_2_points",
]



#for col in chosen04:
#    print(col, data_processed[col].sum())
param_grid = {
    'model__n_estimators': [55,65,75,85,95,100,120,130],
    'model__max_depth': [2,3,4,5],
    'model__learning_rate': [0.1,0.2,0.3]
}

X_train, X_val, y_train, y_val = train_test_split(data_processed[chosen04], data_processed['pass__player_1_won'], random_state=1, test_size=0.20)

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)
prediction = grid_search.predict(X_val)
print(f'Accuracy: {accuracy_score(y_val, prediction)}')

print(grid_search.best_params_)
print(grid_search.best_score_)


model = grid_search.best_estimator_.named_steps['model']

importance = model.feature_importances_

result_import = pd.Series(importance, index=chosen04).sort_values(ascending=False)

print(result_import)


result_perm = permutation_importance(grid_search, X_val, y_val, n_repeats=10, random_state=1)
pd.Series(result_perm.importances_mean, index=chosen04).sort_values(ascending=False)

data_result = pd.DataFrame({
    'mean': result_perm.importances_mean,
    'std': result_perm.importances_std
    }, index=chosen04).sort_values('mean', ascending=False)



print(data_result)
#correlations = data_processed.corr()['pass__player_1_won'].sort_values(ascending=False)
#print(correlations)