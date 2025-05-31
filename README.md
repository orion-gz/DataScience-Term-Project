# Game Hit Prediction Pipeline: User Manual & Specification

This project outlines a pipeline for predicting whether a game will be a "hit" using the RAWG video games dataset. A "hit" is defined as a game with a rating of 4.0 or higher.

The pipeline encompasses data cleaning, feature engineering (including one-hot encoding for genres, developers, and tags), feature scaling, feature selection, and training a Logistic Regression model with hyperparameter tuning via GridSearchCV.

## Project Goal

The primary objective is to develop an optimal model pipeline for predicting game success (hit/non-hit) using the RAWG dataset. This involves systematically experimenting with various preprocessing techniques, feature selection methods, and model hyperparameters, and creating functions to automate this process. This notebook can serve as a practical example of applying a machine learning workflow to game data.

## Dataset

The dataset used is the "RAWG Video Games Dataset," obtainable from sources like Hugging Face Datasets (`atalaydenknalbant/rawg-games-dataset`) or a provided CSV file (`rawg_games_data.csv`).

Features Overview
* `name`: Name of the game.
* `metacritic`: Metacritic score.
* `genres`: Comma-separated string of genres.
* `developers`: Comma-separated string of developers.
* `playtime`: Average playtime in hours.
* `tags`: Comma-separated string of tags.
* `rating`: Original rating score (used to derive the 'hit' target variable).

## Feature Engineering

The preprocessing steps transform the raw data into a feature set suitable for modeling, which typically includes:
* **Numerical Features:**
    * `metacritic`: (Potentially scaled) Metacritic score.
    * `playtime`: (Potentially scaled) Average playtime.
* **Categorical Features (One-Hot Encoded):**
    * `genre_*`: Binary flags for selected top genres (e.g., `genre_action`).
    * `developer_*`: Binary flags for selected top developers (e.g., `dev_nintendo`).
    * `tag_*`: Binary flags for selected top tags (e.g., `tag_singleplayer`).

## Target Variable

* `hit`: A binary variable created from the `rating` column.
* You can also set your own `hit` rating criteria to set threshold (e.g. `preprocessing_rawg_data(.. rating_threshold=3.5)`)
    * 1: Indicates a "hit" game (if `rating >= 4.0`).
    * 0: Indicates a "non-hit" game.

## Functions Usage

The core of this project involves the `preprocessing_rawg_data` function for data preparation and the `run_logistic_regression` function for model training, tuning, and evaluation.

### 1. Helper Method(For preprocessing)

These auxiliary functions are utilized within the main `preprocessing_rawg_data` function. (Based on user's notebook code)

**`get_filled_genres(df_model)`** : Only Filled in the missing values ​​for the genres data used in whole preprocessing step (NOT FILLED WHOLE DATA).
**CAUTION : THIS DATA IS FILLED BY GENERATIVE AI (DO NOT TRUST THIS SOURCE)**

```python
def get_filled_genres(df_model):
  """
  Fills NaN values in the 'genres' column of the DataFrame with a predefined list.
  THIS FUNCTION ONLY FILLS IN THE PRE-FOUND (HARDCODED) DATA.
  It assumes the order of NaNs matches the order of the filled_genres_list.

  @param df_model (DataFrame): RAWG games data, potentially with NaN genres.
  @return df_model (DataFrame): RAWG games data with NaN genres filled.
  """
  # Pre-defined list of genres to fill NaN values.
  filled_genres_list = [
    "Shooter, Arcade",                             # 1. Time Crisis 3
    "Shooter, Arcade",                             # 2. Time Crisis II
    # ... (Full list as in the notebook code)
    "Action, Shooter"                              # 21. WinBack 2: Project Poseidon
  ]
  genres_nan_indices = df_model[df_model['genres'].isna()].index
  n = len(genres_nan_indices)
  genres_filled_data = pd.Series(filled_genres_list)
  df_model.loc[genres_nan_indices[:n], 'genres'] = genres_filled_data.iloc[:n].values
  return df_model
```

**`get_filled_developers(df_model)`** : Only Filled in the missing values ​​for the developers data used in whole preprocessing step (NOT FILLED WHOLE DATA).
**CAUTION : THIS DATA IS FILLED BY GENERATIVE AI (DO NOT TRUST THIS SOURCE)**

```python
def get_filled_developers(df_model):
  """
  Fills NaN values in the 'developers' column of the DataFrame with a predefined list.
  THIS FUNCTION ONLY FILLS IN THE PRE-FOUND (HARDCODED) DATA.
  It assumes the order of NaNs matches the order of the filled_developers_list.

  @param df_model (DataFrame): RAWG games data, potentially with NaN developers.
  @return df_model (DataFrame): RAWG games data with NaN developers filled.
  """
  # Pre-defined list of developers to fill NaN values
  filled_developers_list = [
    "Nintendo EPD, Grezzo", # 1. The Legend of Zelda: Ocarina of Time 3D
    # ... (Full list of 101 developer names as in the notebook code)
    "Fun Labs, Magic Wand Productions"   # 101. Cabelas Dangerous Hunts 2009
  ]
  developers_nan_indices = df_model[df_model['developers'].isna()].index
  n = len(developers_nan_indices)
  developers_filled_data = pd.Series(filled_developers_list)
  df_model.loc[developers_nan_indices[:n], 'developers'] = developers_filled_data.iloc[:n].values
  return df_model
```

**`check_and_remove_wrong_value(df_model)`** : check wrong values for features

```python
def check_and_remove_wrong_value(df_model):
  """
  Identifies and removes rows from the DataFrame that contain "wrong" data
  based on predefined conditions for 'playtime', 'metacritic', and 'rating'.

  @param df_model (DataFrame): RAWG games data to be checked and cleaned.
  @return df_model_cleaned (DataFrame): DataFrame with rows containing wrong data removed.
  """
  all_indices_to_drop = pd.Index([])
  if 'playtime' in df_model.columns and (df_model['playtime'] < 0).any():
    all_indices_to_drop = all_indices_to_drop.union(df_model[df_model['playtime'] < 0].index)
  if 'metacritic' in df_model.columns:
    if (df_model['metacritic'] < 0).any(): all_indices_to_drop = all_indices_to_drop.union(df_model[df_model['metacritic'] < 0].index)
    if (df_model['metacritic'] > 100).any(): all_indices_to_drop = all_indices_to_drop.union(df_model[df_model['metacritic'] > 100].index)
  if 'rating' in df_model.columns: # Assumes 'rating' column exists
    if (df_model['rating'] < 0).any(): all_indices_to_drop = all_indices_to_drop.union(df_model[df_model['rating'] < 0].index)
    if (df_model['rating'] > 5).any(): all_indices_to_drop = all_indices_to_drop.union(df_model[df_model['rating'] > 5].index)
  df_model_cleaned = df_model.drop(all_indices_to_drop).copy()
  return df_model_cleaned
```

**`OHE(df_model_input, col_name, prefix, top_n=0, least_n=0) (Custom OneHotEncoder)`** : change categorical data(in this project e.g. genres, developers, tags..) into binary data

```python
def OHE(df_model_input, col_name, prefix, top_n=0, least_n=0):
  """
  Custom one-hot encodes a specified column containing comma-separated values.
  New binary columns are created based on frequency (top_n or least_n).

  @param df_model_input (DataFrame): Input DataFrame.
  @param col_name (str): The name of the column to be one-hot encoded.
  @param prefix (str): The prefix to use for the new OHE column names.
  @param top_n (int): If > 0, number of top frequent categories to encode.
  @param least_n (int): If top_n is 0 and least_n is specified, minimum frequency for a category to be encoded.
  @return df_model_output (DataFrame): DataFrame with new one-hot encoded columns added.
  """
  print(f"Performing One-Hot Encoding for column: '{col_name}' with prefix: '{prefix}'")
  df_model_output = df_model_input.copy()
  if col_name not in df_model_output.columns:
      print(f"  [Error] Column '{col_name}' not found for OHE.")
      return df_model_output
  c_data_series = df_model_output[col_name].fillna('').astype(str)
  c_data_list = []
  for d_str in c_data_series:
    items = [item.strip() for item in d_str.split(',')]
    for item in items:
      if item: c_data_list.append(item)
  p_c_data_counts = Counter(c_data_list)
  p_c_data_name_list = []
  if top_n > 0:
    most_common_items = p_c_data_counts.most_common(top_n)
    p_c_data_name_list = [c_name for c_name, count in most_common_items]
  elif least_n > 0:
    p_c_data_name_list = [c_name for c_name, count in p_c_data_counts.items() if count >= least_n]
  else:
    p_c_data_name_list = list(p_c_data_counts.keys())
  ohe_columns_data_dict = {}
  for category_name_to_encode in p_c_data_name_list:
    clean_category_name = "".join(filter(str.isalnum, category_name_to_encode.replace(' ', '_')))
    ohe_col_name = f"{prefix}_{clean_category_name.lower()}"[:60]
    current_ohe_col_values = []
    for single_game_categories_str in c_data_series:
      items_in_game = [item.strip() for item in single_game_categories_str.split(',')]
      current_ohe_col_values.append(1 if category_name_to_encode in items_in_game else 0)
    ohe_columns_data_dict[ohe_col_name] = current_ohe_col_values
  df_ohe_cols = pd.DataFrame(ohe_columns_data_dict, index=df_model_output.index)
  df_model_output = pd.concat([df_model_output, df_ohe_cols], axis=1)
  return df_model_output
```

### 2. Main Preprocessing Function
**`preprocessing_rawg_data(raw_df, ...)`**

```python
def preprocessing_rawg_data(
    raw_df,
    rating_threshold = 4.0,
    top_n_genres = 19, 
    top_n_tags = 200,
    developer_min_game_count = 10,
    filled_genres = True,
    filled_developers = True
):
  """
  Main preprocessing function for the RAWG dataset, based on the notebook's logic.
  Handles initial feature selection, missing data imputation, 
  removal of rows with wrong values, target variable ('hit') creation,
  and one-hot encoding of categorical features.
  NOTE: Scaling is NOT performed here and should be done after train/test split.

  @param raw_df (DataFrame): The raw RAWG games dataset.
  @param rating_threshold (float): Rating score criteria for 'hit' definition.
  @param top_n_genres (int): Number of top genres for OHE.
  @param top_n_tags (int): Number of top tags for OHE.
  @param developer_min_game_count (int): Min game count for developer OHE.
  @param filled_genres (bool): Whether to fill missing 'genres' using helper.
  @param filled_developers (bool): Whether to fill missing 'developers' using helper.

  @return X (DataFrame): Preprocessed features, ready for scaling and modeling.
  @return y (Series): Target variable ('hit').
  """
  features_to_select = ['name', 'metacritic', 'genres', 'developers', 'playtime', 'tags']
  target_column_original = 'rating'
  
  try:
    required_cols = features_to_select + [target_column_original]
    if not all(col in raw_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        raise KeyError(f"Missing required columns in raw_df: {missing_cols}")
    df_model = raw_df[required_cols].copy()
  except NameError:
    print("Error: DataFrame 'raw_df' is not defined.")
    return None, None
  except Exception as e:
    print(f"Error during initial DataFrame creation: {e}")
    return None, None

  df_model.dropna(subset=['name', 'metacritic', 'tags'], axis=0, inplace=True)

  if filled_genres:
    df_model = get_filled_genres(df_model)
  else:
    df_model['genres'] = df_model['genres'].fillna('Unknown')

  if filled_developers:
    df_model = get_filled_developers(df_model)
  else:
    df_model['developers'] = df_model['developers'].fillna('Unknown')
  
  df_model['playtime'] = pd.to_numeric(df_model['playtime'], errors='coerce').fillna(df_model['playtime'].median())
  df_model = check_and_remove_wrong_value(df_model)

  df_model['hit'] = 0
  df_model[target_column_original] = pd.to_numeric(df_model[target_column_original], errors='coerce')
  df_model.loc[df_model[target_column_original] >= rating_threshold, 'hit'] = 1
  df_model['hit'] = df_model['hit'].astype(int)
  
  y = df_model['hit'].copy()
  X = df_model.drop(columns=['hit', target_column_original], errors='ignore')

  if 'tags' in X.columns: X = OHE(X, 'tags', prefix='tag', top_n=top_n_tags)
  if 'genres' in X.columns: X = OHE(X, 'genres', prefix='genre', top_n=top_n_genres)
  if 'developers' in X.columns: X = OHE(X, 'developers', prefix='dev', least_n=developer_min_game_count)

  cols_to_drop_from_X = ['name', 'genres', 'developers', 'tags'] 
  X.drop(columns=cols_to_drop_from_X, inplace=True, errors='ignore')
  
  numerical_cols_final_check = ['metacritic', 'playtime']
  for num_feat in numerical_cols_final_check:
      if num_feat in X.columns:
          X[num_feat] = pd.to_numeric(X[num_feat], errors='coerce')
          X[num_feat].fillna(X[num_feat].median(), inplace=True)
  return X, y
```

### 3. Main Modeling Pipeline Function
**`run_logistic_regression(X, y, numerical_feature_names, ...)`**

This function takes the preprocessed data and performs train/test splitting, scaling (optional, via ColumnTransformer), feature selection (optional, via SelectKBest), and Logistic Regression model training with hyperparameter tuning using GridSearchCV.

```python
def run_logistic_regression(
    X,
    y, 
    numerical_feature_names,
    test_size = 0.2,
    random_state = 42,
    selector_instance=SelectKBest(score_func=f_classif),
    model_params_grid=None, 
    cv_fold_count=5,
    main_scoring_metric='f1_weighted'
):
  """
  Builds, tunes, and evaluates a Logistic Regression pipeline using GridSearchCV.
  Handles data splitting, scaling of numerical features, optional feature selection,
  and model hyperparameter tuning.

  @param X (DataFrame): DataFrame of preprocessed features (OHE done).
  @param y (Series): Target variable.
  @param numerical_feature_names (list): List of numerical feature column names in X.
  @param test_size (float): Proportion for the test split.
  @param random_state (int): Seed for random operations.
  @param selector_instance (sklearn.feature_selection selector or None): Feature selector for the pipeline.
  @param model_params_grid (list of dict or dict): Parameter grid for GridSearchCV.
                                                  Should include params for scaler (as 'data_preprocessor__num__scaler'),
                                                  selector (as 'selector__<param>'), and model (as 'model__<param>').
  @param cv_fold_count (int): Number of CV folds.
  @param main_scoring_metric (str): Metric for GridSearchCV optimization.

  @return best_pipeline_model (sklearn.pipeline.Pipeline): Best trained pipeline.
  @return evaluation_metrics_dict (dict): Test set evaluation metrics.
  @return optimal_hyperparameters (dict): Best parameters found.
  """
  print("\n--- Starting Logistic Regression Experiment with GridSearchCV ---")

  # 1. Train-Test Split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=True
  )
  print(f"  Data split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

  # 2. Define pipeline steps
  pipeline_steps = []
  
  # Preprocessor: Scales numerical and passes through OHE features
  ohe_feature_names = [col for col in X_train.columns if col not in numerical_feature_names]
  # Define a numerical transformer pipeline that includes a scaler (to be tuned by GridSearchCV)
  numerical_transformer_pipeline = Pipeline(steps=[('scaler', StandardScaler())]) 
  
  transformers_for_ct = []
  if numerical_feature_names:
      transformers_for_ct.append(('num', numerical_transformer_pipeline, numerical_feature_names))
  if ohe_feature_names:
      transformers_for_ct.append(('cat', 'passthrough', ohe_feature_names))

  if transformers_for_ct:
      data_preprocessor_ct = ColumnTransformer(transformers=transformers_for_ct, remainder='passthrough')
      pipeline_steps.append(('data_preprocessor', data_preprocessor_ct))
  
  # Feature Selection Step
  if selector_instance:
      pipeline_steps.append(('selector', selector_instance))
  
  # Model Step
  pipeline_steps.append(('model', LogisticRegression(random_state=random_state, max_iter=1000))) # Base model
  pipeline = Pipeline(steps=pipeline_steps)

  # 3. GridSearchCV for hyperparameter tuning
  hyperparameter_grid_to_use = model_params_grid
  if not model_params_grid: 
      # Default grid for a single run if no specific grid is provided
      hyperparameter_grid_to_use = {
          'data_preprocessor__num__scaler': [StandardScaler()], # Default scaler
          **({'selector__k': [X_train.shape[1]]} if selector_instance else {}), # Use all features
          'model__solver': ['liblinear'], 
          'model__C': [1.0], 
          'model__class_weight': ['balanced']
      }
      print(f"  [Info] No model_params_grid provided. Using default grid: {hyperparameter_grid_to_use}")

  print(f"  Starting GridSearchCV with param_grid: {hyperparameter_grid_to_use}")
  grid_search_cv = GridSearchCV(
      pipeline, hyperparameter_grid_to_use, 
      cv=cv_fold_count, 
      scoring=main_scoring_metric, 
      n_jobs=-1, # Use all available CPU cores
      verbose=1, # Print progress
      error_score='raise' # Raise error if a combination fails
  )
  
  grid_search_cv.fit(X_train, y_train) # Fit on X_train (pipeline handles internal scaling)
  
  optimized_pipeline_model = grid_search_cv.best_estimator_
  optimal_hyperparameters = grid_search_cv.best_params_
  print(f"  Best CV Score ({main_scoring_metric}): {grid_search_cv.best_score_:.4f}")
  print(f"  Best Parameters: {optimal_hyperparameters}")

  # 4. Evaluate the best model on the Test Set
  y_predict_test = optimized_pipeline_model.predict(X_test)
  y_proba_test = optimized_pipeline_model.predict_proba(X_test)[:, 1] # Probabilities for positive class

  evaluation_metrics_dict = {
      "accuracy": accuracy_score(y_test, y_predict_test),
      "precision": precision_score(y_test, y_predict_test, zero_division=0),
      "recall": recall_score(y_test, y_predict_test, zero_division=0),
      "f1_score": f1_score(y_test, y_predict_test, zero_division=0),
      "roc_auc": roc_auc_score(y_test, y_proba_test),
      "confusion_matrix": confusion_matrix(y_test, y_predict_test).tolist() 
  }

  # Print evaluation results
  print("\n  --- Test Set Evaluation Results ---")
  for metric_name, score_value in evaluation_metrics_dict.items():
      if metric_name != "confusion_matrix":
          print(f"  {metric_name.capitalize()} : {score_value:.4f}")
      else:
          print(f"  {metric_name.capitalize()} :\n{np.array(score_value)}")

  # Plot Confusion Matrix
  plt.figure(figsize=(6,4))
  sns.heatmap(evaluation_metrics_dict["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, 
              xticklabels=['Predicted: Non-Hit (0)', 'Predicted: Hit (1)'], 
              yticklabels=['Actual: Non-Hit (0)', 'Actual: Hit (1)'])
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.title("Confusion Matrix on Test Data (Logistic Regression - Best Tuned)")
  plt.show()

  print("--- Logistic Regression Experiment Complete ---")
  return optimized_pipeline_model, evaluation_metrics_dict, optimal_hyperparameters

How to Use
Load Raw Data:

import pandas as pd
# df_raw = pd.read_csv("path/to/your/rawg_games_data.csv") 
# Example: using the 'df' variable from the notebook's cell 400
df_raw = df 

Run Preprocessing:
This step utilizes the helper functions (get_filled_genres, get_filled_developers, check_and_remove_wrong_value, OHE) internally.

X_processed, y_target = preprocessing_rawg_data(
    raw_df=df_raw
    # Other parameters like top_n_tags can be set here if needed
)
# Define numerical features for scaling (based on columns remaining in X_processed after OHE)
numerical_features = ['metacritic', 'playtime'] 

Define Parameter Grid for run_logistic_regression:
This grid allows GridSearchCV to test different scalers, feature selection k values, and logistic regression hyperparameters.

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

if 'X_processed' in locals():
    param_grid_list_for_lr = [
        { 
            'data_preprocessor__num__scaler': [StandardScaler(), RobustScaler(), MinMaxScaler(), 'passthrough'], 
            'selector__k': [50, 100, 150, X_processed.shape[1]], # Use all features if X_processed is defined
            'model__solver': ['liblinear'],
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.01, 0.1, 1.0, 5.0, 10.0],
            'model__class_weight': ['balanced', {0:1, 1:1.5}, {0:1, 1:2}, {0:1, 1:2.5}, {0:1, 1:3}]
        },
        # Add more dictionaries for other solver/penalty combinations if desired
        # Example for 'saga' solver (supports l1, l2, elasticnet, none for penalty)
        # { 
        #     'data_preprocessor__num__scaler': [StandardScaler(), RobustScaler(), MinMaxScaler(), 'passthrough'],
        #     'selector__k': [50, 100, 150, X_processed.shape[1]],
        #     'model__solver': ['saga'],
        #     'model__penalty': ['l1', 'l2'], # 'elasticnet' requires 'l1_ratio'
        #     'model__C': [0.01, 0.1, 1.0, 5.0, 10.0],
        #     'model__class_weight': ['balanced', {0:1, 1:1.5}, {0:1, 1:2}]
        # }
    ]
else:
    print("X_processed is not defined. Cannot set 'selector__k' dynamically.")
    param_grid_list_for_lr = [] # Or a default grid not dependent on X_processed.shape
```

**Run Logistic Regression Experiment**

```pythob
if 'X_processed' in locals() and 'y_target' in locals() and param_grid_list_for_lr:
    best_lr_model, lr_metrics, lr_params = run_logistic_regression(
        X_processed, 
        y_target,   
        numerical_feature_names=numerical_features, 
        selector_instance=SelectKBest(score_func=f_classif), # Pass the selector instance
        model_params_grid=param_grid_list_for_lr, 
        main_scoring_metric='f1_weighted' # Or 'roc_auc', etc.
    )
    print("\nLogistic Regression - Best Overall Tuned Parameters:", lr_params)
    print("Logistic Regression - Final Test Set Metrics (Tuned):", lr_metrics)
else:
    print("Please ensure 'X_processed', 'y_target', 'numerical_features', and 'param_grid_list_for_lr' are defined.")
```

Dependencies
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
