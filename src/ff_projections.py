import pymc as pm
import arviz as az
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score
import numpy as np
import pandas as pd

def read_in_data_for_projections(filepath, train_min_year=None, train_test_split_year=2023):
    df = pd.read_csv(filepath)
    projection_models_test = df[df['season']>=train_test_split_year]
    projection_models_train = df[df['season']<train_test_split_year]
    if train_min_year:
        projection_models_train = projection_models_train[projection_models_train['season']>=train_min_year]

    projection_models_test = pd.get_dummies(projection_models_test, columns=['position'])
    projection_models_train = pd.get_dummies(projection_models_train, columns=['position'])

    return projection_models_train, projection_models_test

def create_X_y_train_test(pm_train_df, pm_test_df, cols_to_drop=['season', 'gsis_id', 'full_name_all_players', 'fantasy_pts']):
    X_train = pm_train_df.drop(columns=cols_to_drop)
    y_train = pm_train_df[['fantasy_pts']]
    X_test = pm_test_df.drop(columns=cols_to_drop)
    y_test = pm_test_df[['fantasy_pts']]

    preprocessing_pipeline = Pipeline([
        ('imputer', KNNImputer()),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('pca', PCA(n_components=0.95))
    ])

    X_train = preprocessing_pipeline.fit_transform(X_train)
    X_test = preprocessing_pipeline.transform(X_test)

    return X_train, y_train, X_test, y_test

def logistic_regression_target_threshold(X_train, y_train, X_test, y_test, threshold=83.125):
    y_train_binary = y_train <= 83.125
    y_test_binary = y_test <= 83.125

    pipe = Pipeline([
        ('classifier', LogisticRegression())
    ])

    pipe.fit(X_train, y_train_binary)

    y_pred = pipe.predict(X_test)

    accuracy = accuracy_score(y_test_binary, y_pred)
    precision = precision_score(y_test_binary, y_pred)
    recall = recall_score(y_test_binary, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    print(confusion_matrix(y_test_binary, y_pred))

    return y_pred, y_train_binary

def add_is_draftable_column(X_train, y_train, X_test, y_test, threshold=83.125):
    y_pred, y_train_binary = logistic_regression_target_threshold(X_train, y_train, X_test, y_test, threshold=threshold)
    X_test = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])
    X_train = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])

    X_test['is_draftable'] = y_pred.astype(int)
    X_train['is_draftable'] = y_train_binary.astype(int)

    return X_train, X_test

def run_pm_model(X_train, y_train):

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    coef_mean = lr.coef_
    intercept_mean = lr.intercept_
    residuals = y_train - lr.predict(X_train)
    sigma_est = residuals.std()

    print(f"Mean of coefficients: {coef_mean}")
    print(f"Mean of intercept: {intercept_mean}")
    print(f"Standard deviation of residuals: {sigma_est}")

    df = pd.DataFrame(X_train)
    df["target"] = y_train.values

    df_clean = df.dropna()

    X_pm_train = df_clean.drop("target", axis=1).values
    y_pm_train = df_clean["target"].values

    feature_names = df_clean.drop("target", axis=1).columns.tolist()

    with pm.Model(coords={"features": feature_names}) as model:
        X_data = pm.Data("X_data", X_pm_train, dims=("obs", "features"))
        y_data = pm.Data("y_data", y_pm_train, dims="obs")

        intercept = pm.Normal("intercept", mu=intercept_mean, sigma=5)
        betas = pm.Normal("betas", mu=coef_mean, sigma=1.0, dims="features")
        sigma = pm.HalfNormal("sigma", sigma=sigma_est)

        mu = intercept + pm.math.dot(X_data, betas)

        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data, dims="obs")

        trace = pm.sample(draws=2000, tune=2000, chains=4, cores=4, target_accept=0.95, random_seed=11)

    return trace

def create_credible_interval(posterior_pred_samples, interval_size):
    begin = (100 - interval_size)/2
    return np.percentile(posterior_pred_samples, [begin, (100-begin)])

def predict_player(i, trace, X_test, y_test):
    i = 132
    player_features = X_test[i]

    print(player_features)
    print(y_test.iloc[i])

    intercept_samples = trace.posterior["intercept"].values.flatten()
    betas_samples = trace.posterior["betas"].values
    sigma_samples = trace.posterior["sigma"].values.flatten()

    n_chains, n_draws, n_features = betas_samples.shape
    betas_samples = betas_samples.reshape(n_chains * n_draws, n_features)

    mu_samples = intercept_samples + np.dot(betas_samples, player_features)

    posterior_pred_samples = np.random.normal(mu_samples, sigma_samples)

    projected_median = np.median(posterior_pred_samples)
    credible_interval_95 = np.percentile(posterior_pred_samples, [2.5, 97.5]) # Use create credible interval!
    credible_interval_90 = np.percentile(posterior_pred_samples, [5, 95])
    credible_interval_85 = np.percentile(posterior_pred_samples, [7.5, 92.5])
    credible_interval_75 = np.percentile(posterior_pred_samples, [12.5, 87.5])
    credible_interval_50 = np.percentile(posterior_pred_samples, [25, 75])
    prob_gt_200 = np.mean(posterior_pred_samples > 200)

    print(f"Projected season points (median): {projected_median:.1f}")
    print(f"95% credible interval: [{credible_interval_95[0]:.1f}, {credible_interval_95[1]:.1f}]")
    print(f"90% credible interval: [{credible_interval_90[0]:.1f}, {credible_interval_90[1]:.1f}]")
    print(f"85% credible interval: [{credible_interval_85[0]:.1f}, {credible_interval_85[1]:.1f}]")
    print(f"75% credible interval: [{credible_interval_75[0]:.1f}, {credible_interval_75[1]:.1f}]")
    print(f"50% credible interval: [{credible_interval_50[0]:.1f}, {credible_interval_50[1]:.1f}]")
    print(f"Probability points > 200: {prob_gt_200:.2%}")