# A py file for creating fantasy football projections using a probabilistic model

# Import statements
import pymc as pm
import arviz as az
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_in_data_for_projections(filepath, train_min_year=None, train_test_split_year=2023):
    """Reads raw player data from CSV and splits into training and testing sets by season.

    Args:
        filepath (str): Path to the CSV file containing player data.
        train_min_year (int, optional): Minimum year for training data. Defaults to None.
        train_test_split_year (int): Season to split train and test sets. Defaults to 2023.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Training dataset.
            - pd.DataFrame: Testing dataset.
    """
    # Read in the data and split into train and test sets based on the season
    df = pd.read_csv(filepath)
    projection_models_test = df[df['season']>=train_test_split_year]
    projection_models_train = df[df['season']<train_test_split_year]
    if train_min_year:
        # If train_min_year is specified, filter the training data to only include seasons >= train_min_year
        projection_models_train = projection_models_train[projection_models_train['season']>=train_min_year]

    # Get dummy/OHE variables for the 'position' column
    projection_models_test = pd.get_dummies(projection_models_test, columns=['position'])
    projection_models_train = pd.get_dummies(projection_models_train, columns=['position'])

    return projection_models_train, projection_models_test

def create_X_y_train_test(pm_train_df, pm_test_df, cols_to_drop=['season', 'gsis_id', 'full_name_all_players', 'fantasy_pts']):
    """Creates input (X) and output (y) datasets and applies preprocessing.

    This function handles imputation, scaling, polynomial expansion, and dimensionality reduction in a pipeline.

    Args:
        pm_train_df (pd.DataFrame): Training dataset.
        pm_test_df (pd.DataFrame): Testing dataset.
        cols_to_drop (list): List of column names to drop. Defaults to key identifiers and target.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Preprocessed training features.
            - pd.DataFrame: Training targets.
            - np.ndarray: Preprocessed testing features.
            - pd.DataFrame: Testing targets.
    """
    # Create X and y for training and testing
    X_train = pm_train_df.drop(columns=cols_to_drop)
    y_train = pm_train_df[['fantasy_pts']]
    X_test = pm_test_df.drop(columns=cols_to_drop)
    y_test = pm_test_df[['fantasy_pts']]

    preprocessing_pipeline = Pipeline([
        ('imputer', KNNImputer()),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)), # Adding polynomial features helps capture non-linear relationships
        ('pca', PCA(n_components=0.95)) # PCA to reduce dimensionality added from polynomial features
    ])

    # Should attempt training without PCA when compute time available

    # Fit training and transform the training and test sets
    print("Fitting preprocessing pipeline...")
    X_train = preprocessing_pipeline.fit_transform(X_train)
    X_test = preprocessing_pipeline.transform(X_test)

    return X_train, y_train, X_test, y_test

def logistic_regression_target_threshold(X_train, y_train, X_test, y_test, threshold=83.125):
    """Trains a logistic regression model to classify players as draftable or not.

    Uses a fantasy point threshold to create a binary classification label for draftability.

    Args:
        X_train (np.ndarray): Training features.
        y_train (pd.DataFrame): Training target values.
        X_test (np.ndarray): Testing features.
        y_test (pd.DataFrame): Testing target values.
        threshold (float): Threshold for draftability. Defaults to 83.125.

    Returns:
        tuple: Predicted binary labels for the test set, and true binary labels for the training set.
    """
    # Given a threshold, classify players as draftable or not based on their projected fantasy points
    # Threshold is set to 83.125, which is the 25th percentile of fantasy points for last rounders (assuming a 12-team league with 14 rounds)
    y_train_binary = y_train <= 83.125
    y_test_binary = y_test <= 83.125

    pipe = Pipeline([
        ('classifier', LogisticRegression())
    ])

    # Fit the logistic regression model
    print("Fitting logistic regression model...")
    pipe.fit(X_train, y_train_binary)

    y_pred = pipe.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test_binary, y_pred)
    precision = precision_score(y_test_binary, y_pred)
    recall = recall_score(y_test_binary, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_binary, y_pred))

    return y_pred, y_train_binary.values.flatten() # Must flatten y_train_binary to match the shape of y_pred

def add_is_draftable_column(X_train, y_train, X_test, y_test, threshold=83.125):
    """Adds a binary 'is_draftable' feature to both training and test data.

    Args:
        X_train (np.ndarray): Training features.
        y_train (pd.DataFrame): Training target values.
        X_test (np.ndarray): Testing features.
        y_test (pd.DataFrame): Testing target values.
        threshold (float): Fantasy point threshold for draftability.

    Returns:
        tuple: Updated training and testing DataFrames with 'is_draftable' column added.
    """

    y_pred, y_train_binary = logistic_regression_target_threshold(X_train, y_train, X_test, y_test, threshold=threshold)
    X_test = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])
    X_train = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])

    # Add the is_draftable column
    X_test['is_draftable'] = y_pred.astype(int) # Must use predictions from the logistic regression model for test set
    X_train['is_draftable'] = y_train_binary.astype(int) # Can use the binary target from training set

    return X_train, X_test

def run_pm_model(X_train, y_train):
    """Fits a Bayesian linear regression model using PyMC to estimate fantasy point distributions.

    Provides a probabilistic framework to model uncertainty in player fantasy point projections.

    Args:
        X_train (np.ndarray): Preprocessed training feature matrix.
        y_train (np.ndarray): Training targets.

    Returns:
        arviz.InferenceData: Trace object containing posterior samples.
    """
    # Begin by fitting a linear regression model to get initial estimates of priors
    print("Fitting initial linear regression model to get priors...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    coef_mean = np.array(lr.coef_).flatten()
    intercept_mean = lr.intercept_
    residuals = y_train - lr.predict(X_train)
    sigma_est = residuals.std()

    print(f"Mean of coefficients: {coef_mean}")
    print(f"Mean of intercept: {intercept_mean}")
    print(f"Standard deviation of residuals: {sigma_est}")

    # Ensure X_train and y_train are in the proper format
    df = pd.DataFrame(X_train)
    df["target"] = y_train.values

    df_clean = df.dropna()

    X_pm_train = df_clean.drop("target", axis=1).values
    y_pm_train = df_clean["target"].values

    # Obtain feature names for the model (when using PCA, not all that helpful)
    feature_names = df_clean.drop("target", axis=1).columns.tolist()

    # Create a PyMC model
    with pm.Model(coords={"features": feature_names}) as model:
        X_data = pm.Data("X_data", X_pm_train, dims=("obs", "features"))
        y_data = pm.Data("y_data", y_pm_train, dims="obs")

        # Define priors based on the initial linear regression estimates
        intercept = pm.Normal("intercept", mu=intercept_mean, sigma=5)
        betas = pm.Normal("betas", mu=coef_mean, sigma=1.0, dims="features")
        sigma = pm.HalfNormal("sigma", sigma=sigma_est)

        mu = intercept + pm.math.dot(X_data, betas)

        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data, dims="obs")

        # Sample from the posterior
        print("Sampling from the posterior...")
        trace = pm.sample(draws=2000, tune=2000, chains=4, cores=4, target_accept=0.95, random_seed=11)

    return trace

def split_data_and_train_pm_model(filepath, cols_to_drop=['season', 'gsis_id', 'full_name_all_players', 'fantasy_pts'], train_min_year=None, train_test_split_year=2023):
    """Orchestrates the entire pipeline from reading raw data to fitting a Bayesian model.

    Args:
        filepath (str): Path to the input CSV data.
        cols_to_drop (list): Columns to exclude from features. Defaults to key identifiers and target.
        train_min_year (int, optional): Minimum training year filter.
        train_test_split_year (int): Season year to separate train and test data.

    Returns:
        tuple: A tuple containing:
            - arviz.InferenceData: Posterior samples trace.
            - pd.DataFrame: Preprocessed test features with draftability flag.
            - pd.DataFrame: Ground-truth test target values.
    """
    # Create train and test datasets
    pm_train_df, pm_test_df = read_in_data_for_projections(filepath, train_min_year=train_min_year, train_test_split_year=train_test_split_year)
    # Create X and y for training and testing
    X_train, y_train, X_test, y_test = create_X_y_train_test(pm_train_df, pm_test_df, cols_to_drop=cols_to_drop)
    # Add is_draftable column based on logistic regression threshold
    X_train, X_test = add_is_draftable_column(X_train, y_train, X_test, y_test)
    # Run the probabilistic model
    print("Running probabilistic model...")
    trace = run_pm_model(X_train.values, y_train)

    return trace, X_test, y_test

def create_credible_interval(posterior_pred_samples, interval_size):
    begin = (100 - interval_size)/2
    return np.percentile(posterior_pred_samples, [begin, (100-begin)])

def predict_player(i, trace, X_test, y_test, plot=True):
    """Predicts and visualizes the posterior distribution of fantasy points for a single player.

    Args:
        i (int): Index of the player in the test set.
        trace (arviz.InferenceData): Posterior samples trace from PyMC model.
        X_test (pd.DataFrame): Test feature DataFrame.
        y_test (pd.DataFrame): True fantasy points for test set.
        plot (bool): Whether to display a histogram of the posterior predictive distribution.

    Returns:
        None
    """
    # Identify the player features and true fantasy points for the i-th player in the test set
    player_features = X_test.iloc[i]
    print(player_features)
    print(y_test.iloc[i])

    # Extract the posterior samples from the trace
    intercept_samples = trace.posterior["intercept"].values.flatten()
    betas_samples = trace.posterior["betas"].values
    sigma_samples = trace.posterior["sigma"].values.flatten()

    # Reshape the betas_samples to match the player features
    n_chains, n_draws, n_features = betas_samples.shape
    betas_samples = betas_samples.reshape(n_chains * n_draws, n_features)

    # Calculate the posterior predictive distribution
    mu_samples = intercept_samples + np.dot(betas_samples, player_features)

    # Take mean and std of the posterior predictive distribution to create samples
    posterior_pred_samples = np.random.normal(mu_samples, sigma_samples)

    # Calculate statistics from the posterior predictive samples
    projected_median = np.median(posterior_pred_samples)
    credible_interval_95 = create_credible_interval(posterior_pred_samples, 95)
    credible_interval_90 = create_credible_interval(posterior_pred_samples, 90)
    credible_interval_85 = create_credible_interval(posterior_pred_samples, 85)
    credible_interval_75 = create_credible_interval(posterior_pred_samples, 75)
    credible_interval_50 = create_credible_interval(posterior_pred_samples, 50)
    prob_gt_200 = np.mean(posterior_pred_samples > 200)

    print(f"Projected season points (median): {projected_median:.1f}")
    print(f"95% credible interval: [{credible_interval_95[0]:.1f}, {credible_interval_95[1]:.1f}]")
    print(f"90% credible interval: [{credible_interval_90[0]:.1f}, {credible_interval_90[1]:.1f}]")
    print(f"85% credible interval: [{credible_interval_85[0]:.1f}, {credible_interval_85[1]:.1f}]")
    print(f"75% credible interval: [{credible_interval_75[0]:.1f}, {credible_interval_75[1]:.1f}]")
    print(f"50% credible interval: [{credible_interval_50[0]:.1f}, {credible_interval_50[1]:.1f}]")
    print(f"Probability points > 200: {prob_gt_200:.2%}")

    # Plot posterior predictive distribution
    if plot:
        plt.figure(figsize=(10, 6))
        sns.histplot(posterior_pred_samples, bins=50, kde=True, color="skyblue")
        plt.axvline(projected_median, color="red", linestyle="--", label=f"Median: {projected_median:.1f}")
        plt.axvline(200, color="green", linestyle=":", label="200-point threshold")
        plt.title(f"Posterior Predictive Distribution for Player {i}")
        plt.xlabel("Predicted Season Points")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()

    
def create_probabilistic_predictions(trace, pm_test, X_test, save_to_csv=False, filepath=None):
    """
    Generates probabilistic predictions and credible intervals for each player using the fitted PyMC model.

    Args:
        trace (arviz.InferenceData): The trace object containing posterior samples from the PyMC model.
        pm_test (pd.DataFrame): Original test DataFrame including player metadata (e.g., names).
        X_test (pd.DataFrame): Preprocessed test feature matrix.
        save_to_csv (bool, optional): Whether to save the predictions DataFrame as a CSV. Defaults to False.
        filepath (str, optional): Filepath to save CSV if `save_to_csv` is True. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing player names, median predictions, credible intervals,
                      and probabilities of exceeding fantasy point thresholds.
    """
    # returns a df of median predictions and credible intervals for each player in the test set
    median_predictions = []
    credible_intervals_95 = []
    credible_intervals_85 = []
    credible_intervals_50 = []
    all_probabilities = []

    for i in range(len(pm_test)):
        player_features = X_test.iloc[i]

        # Extract the posterior samples from the trace
        intercept_samples = trace.posterior["intercept"].values.flatten()
        betas_samples = trace.posterior["betas"].values
        sigma_samples = trace.posterior["sigma"].values.flatten()

        n_chains, n_draws, n_features = betas_samples.shape
        betas_samples = betas_samples.reshape(n_chains * n_draws, n_features)

        mu_samples = intercept_samples + np.dot(betas_samples, player_features)
        posterior_pred_samples = np.random.normal(mu_samples, sigma_samples)

        median_predictions.append(np.median(posterior_pred_samples))
        credible_intervals_95.append(create_credible_interval(posterior_pred_samples, 95))
        credible_intervals_85.append(create_credible_interval(posterior_pred_samples, 85))
        credible_intervals_50.append(create_credible_interval(posterior_pred_samples, 50))
        prob_gt_300 = np.mean(posterior_pred_samples > 300)
        prob_gt_200 = np.mean(posterior_pred_samples > 200)
        prob_gt_150 = np.mean(posterior_pred_samples > 150)
        prob_gt_100 = np.mean(posterior_pred_samples > 100)
        probabilities = [prob_gt_300, prob_gt_200, prob_gt_150, prob_gt_100]
        all_probabilities.append(probabilities)

    all_probabilities = np.array(all_probabilities)
    # Create a DataFrame with the predictions and credible intervals
    predictions_df = pd.DataFrame({
        'player_name': pm_test['full_name_all_players'].values,
        'median_prediction': median_predictions,
        'credible_interval_95_lower': [ci[0] for ci in credible_intervals_95],
        'credible_interval_95_upper': [ci[1] for ci in credible_intervals_95],
        'credible_interval_85_lower': [ci[0] for ci in credible_intervals_85],
        'credible_interval_85_upper': [ci[1] for ci in credible_intervals_85],
        'probability_gt_300': all_probabilities[:, 0],
        'probability_gt_200': all_probabilities[:, 1]
    })

    if save_to_csv:
        predictions_df.to_csv(filepath, index=False)

    return predictions_df