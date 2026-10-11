"""Random Forest experiment for the California housing price project.

Self-contained: loads housing.csv, rebuilds the feature-engineering and
outlier-removal steps from the EDA notebook, then trains, cross-validates
and grid-searches a RandomForestRegressor and evaluates it on a held-out
test set.

Outputs:
    rf_feature_importances.png - feature importance bar chart of the tuned model
"""

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Feature engineering steps (thresholds copied from the EDA notebook)
# ----------------------------------------------------------------------
class FeaturesAdder(BaseEstimator, TransformerMixin):
    """Adds rooms_per_household, bedrooms_per_room and population_per_household."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
        X['population_per_household'] = X['population'] / X['households']
        return X


class RemoveOutliers(BaseEstimator, TransformerMixin):
    """Removes outliers using thresholds fixed during EDA."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[(X['median_house_value'] != 500001) | (X['median_income'] >= 2)]
        X = X[X['median_income'] <= 11]
        X = X[(X['median_house_value'] != 350000) | (X['median_income'] >= 1.5)]
        X = X[(X['median_house_value'] != 450000) | (X['median_income'] >= 2)]
        X = X[(X['median_house_value'] >= 350000) | (X['median_income'] <= 9.5)]
        X = X[X['population'] <= 9000]
        X = X[(X['population_per_household'] >= 1.15) & (X['population_per_household'] <= 6.5)]
        X = X[X['rooms_per_household'] < 20]
        X = X[X['bedrooms_per_room'] < 0.5]
        return X.reset_index(drop=True)


class FeaturesTransformer(BaseEstimator, TransformerMixin):
    """Power/log transforms skewed numeric features (exponents fixed during EDA)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['total_rooms'] = X['total_rooms'].apply(lambda x: boxcox1p(x, 0.25))
        X['total_bedrooms'] = X['total_bedrooms'].apply(lambda x: boxcox1p(x, 0.25))
        X['households'] = X['households'].apply(lambda x: boxcox1p(x, 0.2))
        X['population'] = X['population'].apply(lambda x: boxcox1p(x, 0.3))
        X['rooms_per_household'] = X['rooms_per_household'].apply(lambda x: np.log1p(x) ** 0.5)
        X['bedrooms_per_room'] = X['bedrooms_per_room'].apply(lambda x: np.log1p(x) ** 0.25)
        X['median_income'] = X['median_income'].apply(lambda x: np.log1p(x) ** 1.25)
        X['population_per_household'] = X['population_per_household'].apply(lambda x: np.log1p(x))
        return X


NUM_FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
]
CAT_FEATURES = ['ocean_proximity']


def load_and_split(csv_path):
    """Stratified 80/20 split on an income category, as in the EDA notebook."""
    df = pd.read_csv(csv_path)
    df['income_cat'] = np.clip(np.ceil(df['median_income'] / 1.5), 0, 5)
    train, test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['income_cat']
    )
    return train.drop('income_cat', axis=1), test.drop('income_cat', axis=1)


def prepare(part, remove_outliers):
    """Feature engineering for one split; returns (features, labels)."""
    part = FeaturesAdder().fit_transform(part)
    if remove_outliers:  # only the training split is filtered, as in the notebook
        part = RemoveOutliers().fit_transform(part)
    labels = part['median_house_value'].copy()
    part = part.drop('median_house_value', axis=1)
    part = FeaturesTransformer().fit_transform(part)
    # ISLAND has ~5 rows; merge it into NEAR BAY like the notebook's GetDummies
    part['ocean_proximity'] = part['ocean_proximity'].replace('ISLAND', 'NEAR BAY')
    return part, labels


def build_preprocessor():
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUM_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES),
    ])


def cv_report(model, X, y, name):
    r2 = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='r2')
    rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, n_jobs=-1,
                                    scoring='neg_mean_squared_error'))
    print(f'{name}: CV R2 = {r2.mean():.4f} +/- {r2.std():.4f}, '
          f'CV RMSE = ${rmse.mean():,.0f}')


def main():
    t0 = time.time()

    train_raw, test_raw = load_and_split('housing.csv')
    X_train, y_train = prepare(train_raw, remove_outliers=True)
    X_test, y_test = prepare(test_raw, remove_outliers=False)
    print(f'Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows '
          f'({time.time() - t0:.1f}s)')

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    feature_names = [n.split('__', 1)[1] for n in preprocessor.get_feature_names_out()]

    print('\n=== Baseline RandomForestRegressor (default parameters) ===')
    baseline = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    cv_report(baseline, X_train, y_train, 'baseline')

    print('\n=== Grid search ===')
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [12, 20, None],
        'min_samples_split': [2, 10],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid, scoring='r2', cv=3, n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f'Best params: {grid.best_params_}')
    print(f'Best grid CV R2: {grid.best_score_:.4f}')

    best = grid.best_estimator_
    print('\n=== Tuned model, 5-fold CV on train ===')
    cv_report(best, X_train, y_train, 'tuned  ')

    print('\n=== Final evaluation on held-out test set ===')
    best.fit(X_train, y_train)
    train_pred = best.predict(X_train)
    test_pred = best.predict(X_test)
    print(f'Train RMSE: ${np.sqrt(mean_squared_error(y_train, train_pred)):,.0f}   '
          f'R2: {r2_score(y_train, train_pred):.4f}')
    print(f'Test  RMSE: ${np.sqrt(mean_squared_error(y_test, test_pred)):,.0f}   '
          f'R2: {r2_score(y_test, test_pred):.4f}   '
          f'MAE: ${mean_absolute_error(y_test, test_pred):,.0f}')

    importances = pd.Series(best.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    print('\nTop feature importances:')
    for name, val in importances.head(10).items():
        print(f'  {name:<28} {val:.4f}')

    fig, ax = plt.subplots(figsize=(9, 6))
    importances.plot.bar(ax=ax)
    ax.set_title('Random Forest feature importances (tuned model)')
    ax.set_ylabel('Importance')
    fig.tight_layout()
    fig.savefig('rf_feature_importances.png', dpi=150)
    print(f'\nSaved rf_feature_importances.png  (total {time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
