import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def get_filled_genres(
    df_model
):
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
    "Casual, Family",                              # 3. EyeToy: Play 2 (Party game elements considered)
    "RPG, Simulation",                             # 4. Monster Rancher 4
    "Action, Shooter, Arcade",                     # 5. Skygunner
    "Action, Adventure, RPG, Simulation",          # 6. Steambot Chronicles
    #"Action, Indie, Platformer, Arcade",          # 7. Aerial_Knights Never Yield (originally commented out)
    "Action, Adventure",                           # 8. Monster House (this would be index 6 if above is uncommented)
    "Action, Adventure, Platformer",               # 9. Brave: A Warrior's Tale (this would be index 7)
    "Action, Platformer, Shooter",                 # 10. Megaman X8 (Mega Man X8)
    "Racing, Casual, Family",                      # 11. Madagascar Kartz (DS)
    "Strategy, RPG",                               # 12. Heroes of Mana
    "Action, Shooter",                             # 13. Gungrave
    "Action, Adventure, Shooter",                  # 14. Resident Evil: Dead Aim
    "Strategy, RPG",                               # 15. Chaos Wars
    "Action, RPG",                                 # 16. Shining Force EXA
    "Action, Fighting, Platformer",                # 17. Altered Beast: Guardian of the Realms
    "Shooter, Arcade",                             # 18. Time Crisis 4
    "Action, RPG",                                 # 19. Evergrace
    #"Action, Adventure, Platformer",              # 20. The Legend of Spyro: Dawn of the Dragon (DS) (originally commented out)
    "Action, Shooter"                              # 21. WinBack 2: Project Poseidon (this would be index 18)
  ]

  # Get the indices of rows where 'genres' data is NaN
  genres_nan_indices = df_model[df_model['genres'].isna()].index
  # Number of NaN entries to fill
  n = len(genres_nan_indices)

  # Convert the hardcoded list to a Pandas Series
  # Ensures that only up to 'n' items from filled_genres_list are used if n is smaller
  genres_filled_data = pd.Series(filled_genres_list)

  # Inserts each value from 'genres_filled_data' into 'df_model' at the NaN indices.
  # .iloc[:n] ensures that we only try to use as many filled values as there are NaNs,
  # or as many as are available in genres_filled_data if that's shorter.
  # .values is used to assign raw values, avoiding potential index mismatch issues with Series assignment.
  df_model.loc[genres_nan_indices[:n], 'genres'] = genres_filled_data.iloc[:n].values

  return df_model

def get_filled_developers(
    df_model
):
  """
  Fills NaN values in the 'developers' column of the DataFrame with a predefined list.
  THIS FUNCTION ONLY FILLS IN THE PRE-FOUND (HARDCODED) DATA.
  It assumes the order of NaNs matches the order of the filled_developers_list.

  @param df_model (DataFrame): RAWG games data, potentially with NaN developers.
  @return df_model (DataFrame): RAWG games data with NaN developers filled.
  """

  # Pre-defined list of developers to fill NaN values
  filled_developers_list = [
    "Nintendo EPD, Grezzo", "Blizzard Entertainment", "Blizzard Entertainment", "EA Canada",
    "Blizzard Entertainment", "Team Ninja, Tecmo", "Blizzard Entertainment, Iron Galaxy",
    "Nintendo EAD", "Blizzard Entertainment", "Retro Studios", "Nintendo EPD, Grezzo",
    "Monolith Soft", "Bizarre Creations", "Blizzard Entertainment", "DrinkBox Studios",
    "Blizzard Entertainment", "Blizzard Entertainment", "Nintendo EPD", "Game Freak",
    "Blizzard Entertainment", "Blizzard Entertainment", "Blizzard Entertainment",
    "Monolith Soft", "Blizzard Entertainment", "EA Vancouver", "Sports Interactive",
    "Bounding Box Software Inc.", "Barnhouse Effect", "NeocoreGames", "Neverland",
    "Blizzard Entertainment", "Bandai Namco Studios", "Gameloft", "Treyarch",
    "Mekensleep", "Dimps, Sonic Team", "NeocoreGames", "Team Pesky", "I-Illusions",
    "Nintendo EPD", "Fantasy Flight Interactive", "Kuju Entertainment", "Psikyo",
    "Matrix Software, IronNos", "Bungie", "Ubisoft Paris", "Sonic Team USA",
    "VooFoo Studios", "Ubisoft Paris, Ubisoft Montpellier", "NeocoreGames",
    "Access Games, Project Aces", "Access Games, Project Aces", "BEC (Bandai Entertainment Company)",
    "Metronomik", "Reality Pump Studios", "NeocoreGames", "Capcom",
    "EA Montreal, Exient Entertainment, EA Black Box", "Nintendo EPD", "Acquire, FromSoftware",
    "SCE Studio Foster City", "S-GAME (Soulframe)", "Cavia", "Acclaim Studios Teesside",
    "Q-Games, Double Eleven", "Konami Computer Entertainment Japan (KCEJ)", "Jupiter Corporation",
    "Deep Red Games", "Office Create (Cooking Mama Limited)", "Sega Sports R&D (Sega Sports Japan)",
    "SCE Japan Studio", "Gaming Minds Studios", "Venom Games", "Cavia, Capcom Production Studio 3",
    "inXile Entertainment", "Konami, Tenky", "NeocoreGames",
    "KCEO (Konami Computer Entertainment Osaka)", "Cavia", "Now Production",
    "Reality Pump Studios", "Climax Studios", "SuperVillain Studios",
    "Blizzard Entertainment, Lemon Sky Studios", "Cavia, Marvelous", "Now Production",
    "Sonic Team", "Ertain, Koei", "NeocoreGames", "Sonic Team, Sega", "FreeStyleGames",
    "Rebellion Developments", "SCE Japan Studio, Shift", "Ubisoft Paris", "Cavia, Koei",
    "V7 Entertainment", "Ubisoft Reflections, Ubisoft Romania", "WorkJam, Aksys Games",
    "Wizarbox", "Gearbox Software, Ubisoft Montreal (Wii)", "Fun Labs, Magic Wand Productions"
  ]

  # Get the indices of rows where 'developers' data is NaN
  developers_nan_indices = df_model[df_model['developers'].isna()].index
  # Number of NaN entries to fill
  n = len(developers_nan_indices)

  # Convert the hardcoded list to a Pandas Series
  developers_filled_data = pd.Series(filled_developers_list)

  # Inserts each value from 'developers_filled_data' into 'df_model' at the NaN indices.
  # .iloc[:n] ensures correct length if 'n' is less than len(developers_filled_data).
  # .values is used for raw value assignment.
  df_model.loc[developers_nan_indices[:n], 'developers'] = developers_filled_data.iloc[:n].values

  return df_model

def check_and_remove_wrong_value(
    df_model
):
  """
  Identifies and removes rows from the DataFrame that contain "wrong" data
  based on predefined conditions for 'playtime', 'metacritic', and 'rating'.

  @param df_model (DataFrame): RAWG games data to be checked and cleaned.
  @return df_model_cleaned (DataFrame): DataFrame with rows containing wrong data removed.
  """
  # List to store all indices of rows that need to be dropped
  all_indices_to_drop = pd.Index([])

  # Condition 1: playtime cannot be negative
  if 'playtime' in df_model.columns:
    condition1 = df_model['playtime'] < 0
    if condition1.any(): # Check if any such rows exist
      all_indices_to_drop = all_indices_to_drop.union(df_model[condition1].index)

  # Condition 2: metacritic score cannot be negative
  if 'metacritic' in df_model.columns:
    condition2 = df_model['metacritic'] < 0
    if condition2.any():
      all_indices_to_drop = all_indices_to_drop.union(df_model[condition2].index)

  # Condition 3: metacritic score cannot be over 100
  if 'metacritic' in df_model.columns:
    condition3 = df_model['metacritic'] > 100
    if condition3.any():
      all_indices_to_drop = all_indices_to_drop.union(df_model[condition3].index)

  # Condition 4: Rating score cannot be negative (assuming 'rating' column still exists at this point)
  if 'rating' in df_model.columns:
    condition4 = df_model['rating'] < 0
    if condition4.any():
      all_indices_to_drop = all_indices_to_drop.union(df_model[condition4].index)

  # Condition 5: Rating score cannot be over 5
  if 'rating' in df_model.columns:
    condition5 = df_model['rating'] > 5
    if condition5.any():
      all_indices_to_drop = all_indices_to_drop.union(df_model[condition5].index)

  # Drop the identified rows from the DataFrame
  # .copy() is used to ensure df_model_cleaned is a new DataFrame
  df_model_cleaned = df_model.drop(all_indices_to_drop).copy()

  return df_model_cleaned


def OHE(
    df_model_input,
    col_name,
    prefix,
    top_n = 0,
    least_n = 0
):
  """
  Custom one-hot encodes a specified column in the DataFrame.
  The column is expected to contain comma-separated string values.
  New binary columns are created and concatenated to the original DataFrame.

  @param df_model_input (DataFrame): Input DataFrame.
  @param col_name (str): The name of the column to be one-hot encoded.
  @param prefix (str): The prefix to use for the new OHE column names.
  @param top_n (int): If > 0, number of top frequent categories to encode.
  @param least_n (int): If > 0 and top_n is 0, minimum frequency for a category to be encoded.
  @return df_model_output (DataFrame): DataFrame with new one-hot encoded columns.
  """
  print(f"Performing One-Hot Encoding for column: '{col_name}' with prefix: '{prefix}'")
  df_model_output = df_model_input.copy()

  # Ensure the column exists and handle potential NaNs by converting to string
  if col_name not in df_model_output.columns:
      print(f"  [Error] Column '{col_name}' not found for OHE.")
      return df_model_output

  # c_ means categorical data
  # Ensure all entries are strings and handle NaNs before splitting
  c_data_series = df_model_output[col_name].fillna('').astype(str)
  c_data_list = []

  for d_str in c_data_series:
    # Split by comma, then strip spaces
    items = [item.strip() for item in d_str.split(',')]
    for item in items:
      # Avoid adding empty strings if there are trailing commas or empty entries
      if item:
        c_data_list.append(item)

  # p_ means preprocessed data
  p_c_data_counts = Counter(c_data_list)
  # List of category names to be encoded
  p_c_data_name_list = []

  if top_n > 0:
    # Get list of (category, count) tuples for top_n most common
    most_common_items = p_c_data_counts.most_common(top_n)
    p_c_data_name_list = [c_name for c_name, count in most_common_items] # Get just the names
  elif least_n > 0:
    # Select categories that appear at least 'least_n' times
    p_c_data_name_list = [c_name for c_name, count in p_c_data_counts.items() if count >= least_n]
  else:
    # If neither top_n nor least_n is specified, encode all unique categories found
    p_c_data_name_list = list(p_c_data_counts.keys())

  # Dictionary to hold data for new OHE columns
  ohe_columns_data_dict = {}

  for category_name_to_encode in p_c_data_name_list: # e.g., category_name_to_encode is 'Action'
    # Format the new OHE column name
    # Replaces spaces with underscores, converts to lowercase. Also remove other special chars.
    clean_category_name = "".join(filter(str.isalnum, category_name_to_encode.replace(' ', '_')))
    ohe_col_name = f"{prefix}_{clean_category_name.lower()}"[:60] # Limit col name length

    # Create list of 0s and 1s for the current OHE column
    current_ohe_col_values = []
    for single_game_categories_str in c_data_series: # Iterate through each game's category string
      # Split the game's category string into a list of individual categories
      items_in_game = [item.strip() for item in single_game_categories_str.split(',')]
      if category_name_to_encode in items_in_game:
        current_ohe_col_values.append(1)
      else:
        current_ohe_col_values.append(0)
    ohe_columns_data_dict[ohe_col_name] = current_ohe_col_values

  # Create a DataFrame from the OHE columns data
  df_ohe_cols = pd.DataFrame(ohe_columns_data_dict, index=df_model_output.index)

  # Concatenate the new OHE columns with the output DataFrame
  df_model_output = pd.concat([df_model_output, df_ohe_cols], axis=1)

  return df_model_output


# define preprocessing RAWG games data
def preprocessing_rawg_data(
    raw_df,
    rating_threshold = 4.0,
    top_n_genres = 20,
    top_n_tags = 200,
    developer_min_game_count = 10,
    filled_genres = True,
    filled_developers = True
):
  """
  Main preprocessing function for the RAWG dataset.
  Handles selection of initial features, missing data, target variable creation,
  and one-hot encoding of categorical features ('tags', 'genres', 'developers').
  Scaling should be performed separately after train/test split.

  @param raw_df (DataFrame): The raw RAWG games dataset.
  @param rating_threshold (float): Rating score criteria to determine if a game is a 'hit'.
  @param top_n_genres (int): Number of top genres to use for OHE.
  @param top_n_tags (int): Number of top tags to use for OHE.
  @param developer_min_game_count (int): Minimum game count for a developer to be OHE'd.
  @param filled_genres (bool): Whether to fill missing 'genres' using the helper function.
  @param filled_developers (bool): Whether to fill missing 'developers' using the helper function.

  @return X (DataFrame): Preprocessed features, ready for scaling and modeling.
  @return y (Series): Target variable ('hit').
  """
  # Define initial features and target column
  features_to_select = ['name', 'metacritic', 'genres', 'developers', 'playtime', 'tags']
  target_column_original = 'rating'

  # --- Load DataSet ---
  # Create a working copy of the relevant part of the DataFrame
  # Try-except block for robust data loading
  try:
    # Ensure all necessary columns are present in raw_df before selection
    required_cols = features_to_select + [target_column_original]
    for col in required_cols:
        if col not in raw_df.columns:
            print(f"Error: Column '{col}' not found in raw_df.")
            # raw_df = pd.read_csv("hf://datasets/atalaydenknalbant/rawg-games-dataset/rawg_games_data.csv")
            # df_model = raw_df[required_cols].copy() 
            raise KeyError(f"Column '{col}' not found in raw_df.")
    df_model = raw_df[required_cols].copy()
  except NameError:
    print("Error: DataFrame 'raw_df' is not defined prior to calling this function.")
    return None, None
  except Exception as e:
    print(f"Error during initial DataFrame creation: {e}")
    return None, None

  # --- Cleaning Dirty Data ---
  # Drop rows where 'name', 'metacritic', or 'tags' are NaN
  df_model.dropna(subset=['name', 'metacritic', 'tags'], axis=0, inplace=True)

  # --- Fill Missing Values ---
  # Fill missing genres and developers if flags are True
  if filled_genres:
    df_model = get_filled_genres(df_model)
  else:
    df_model['genres'].dropna()

  if filled_developers:
    df_model = get_filled_developers(df_model)
  else:
    df_model['developers'].dropna()

  # --- Cleaning Wrong Data ---
  # Check and remove rows with "wrong" numerical values
  df_model = check_and_remove_wrong_value(df_model)

  # --- Feature Creation ---
  # Create binary target variable 'hit'
  df_model['hit'] = 0 
  df_model[target_column_original] = pd.to_numeric(df_model[target_column_original], errors='coerce')
  df_model.loc[df_model[target_column_original] >= rating_threshold, 'hit'] = 1
  df_model['hit'] = df_model['hit'].astype(int)

  # Separate target variable y
  y = df_model['hit'].copy()

  # Prepare feature set X by initially dropping target and original rating column
  X = df_model.drop(columns=['hit', target_column_original], errors='ignore')

  # One-Hot Encode 'tags', 'genres', 'developers' using the custom OHE function
  # The OHE will add new columns to X
  if 'tags' in X.columns:
    X = OHE(X, 'tags', prefix='tag', top_n=top_n_tags)

  if 'genres' in X.columns:
    # If top_n_genres is passed as e.g. 19, it will take top 19.
    # If top_n_genres is None, OHE uses all unique.
    X = OHE(X, 'genres', prefix='genre', top_n=top_n_genres)

  if 'developers' in X.columns:
    X = OHE(X, 'developers', prefix='dev', least_n=developer_min_game_count)

  # Drop original text columns and 'name' from the feature set X
  cols_to_drop_from_X = ['name', 'genres', 'developers', 'tags']
  X.drop(columns=cols_to_drop_from_X, inplace=True, errors='ignore')

  return X, y


def run_logistic_regression(
    X,                       
    y,                       
    numerical_feature_names, 
    test_size=0.2,
    random_state=42,
    selector_instance=SelectKBest(score_func=f_classif),
    model_params_grid=None,
    cv_fold_count=5,
    main_scoring_metric='f1_weighted'
):
  """
  Builds, tunes, and evaluates a Logistic Regression pipeline using GridSearchCV.
  This function takes preprocessed features (X) and a target (y), 
  applies train/test split, creates a ColumnTransformer for scaling numerical features,
  optionally performs feature selection, and then tunes/trains a Logistic Regression model.

  @param X (DataFrame): DataFrame of preprocessed features. OHE should be done.
  @param y (Series): Target variable.
  @param numerical_feature_names (list): List of column names in X that are numerical.
  @param test_size (float): Proportion of the dataset to include in the test split.
  @param random_state (int): Seed for the random number generator.
  @param selector_instance (sklearn.feature_selection selector or None): Feature selector to use. Default is SelectKBest(f_classif).
  @param model_params_grid (list of dict or dict): Parameter grid for GridSearchCV. 
  @param cv_fold_count (int): Number of cross-validation folds for GridSearchCV.
  @param main_scoring_metric (str): Scoring metric for GridSearchCV optimization.

  @return best_pipeline_model (sklearn.pipeline.Pipeline): The best trained pipeline from GridSearchCV.
  @return evaluation_metrics_dict (dict): Evaluation metrics on the test set using the best pipeline.
  @return optimal_hyperparameters (dict): Best parameters found by GridSearchCV.
  """
  print("\n--- Starting Logistic Regression Experiment with GridSearchCV ---")

  # Train-Test Split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y,
      test_size=test_size,
      random_state=random_state,
      stratify=y, 
      shuffle=True
  )
  print(f"Data split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

  # Define preprocessor (ColumnTransformer)
  # This will scale numerical features and passthrough others (OHE features)
  ohe_feature_names = [col for col in X_train.columns if col not in numerical_feature_names]
  
  # Define a numerical transformer pipeline that includes a scaler
  # The actual scaler instance will be set by GridSearchCV via 'data_preprocessor__num__scaler'
  numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

  transformers_for_ct = []
  if numerical_feature_names:
      transformers_for_ct.append(('num', numerical_transformer, numerical_feature_names))
  if ohe_feature_names:
      transformers_for_ct.append(('cat', 'passthrough', ohe_feature_names))

  data_preprocessor = ColumnTransformer(
      transformers=transformers_for_ct,
      remainder='passthrough' 
  )

  # Define full pipeline steps
  pipeline_steps = [
      ('data_preprocessor', data_preprocessor)
  ]
  if selector_instance:
      pipeline_steps.append(('selector', selector_instance))
  
  pipeline_steps.append(('model', LogisticRegression(random_state=random_state)))
  pipeline = Pipeline(steps=pipeline_steps)

  # GridSearchCV
  hyperparameter_grid = model_params_grid


  print(f"  Starting GridSearchCV with param_grid: {hyperparameter_grid}")
  grid_search_cv = GridSearchCV(
      pipeline, 
      hyperparameter_grid, 
      cv=cv_fold_count, 
      scoring=main_scoring_metric, 
      n_jobs=-1, 
      verbose=1, 
      error_score='raise'
  )
  
  # Fit on the original X_train (not yet scaled by this function)
  grid_search_cv.fit(X_train, y_train) 
  
  optimized_pipeline_model = grid_search_cv.best_estimator_
  optimal_hyperparameters = grid_search_cv.best_params_
  print(f"  Best CV Score ({main_scoring_metric}): {grid_search_cv.best_score_:.4f}")
  print(f"  Best Parameters: {optimal_hyperparameters}")

  # Evaluate the best model on the Test Set
  y_predict_test = optimized_pipeline_model.predict(X_test)
  y_proba_test = optimized_pipeline_model.predict_proba(X_test)[:, 1]

  evaluation_metrics_dict = {
      "정확도": accuracy_score(y_test, y_predict_test),
      "정밀도": precision_score(y_test, y_predict_test, zero_division=0),
      "Recall": recall_score(y_test, y_predict_test, zero_division=0),
      "F1 Score": f1_score(y_test, y_predict_test, zero_division=0),
      "ROC AUC": roc_auc_score(y_test, y_proba_test),
      "Confusion Matrix": confusion_matrix(y_test, y_predict_test).tolist() 
  }

  print("\n--- Test Set Evaluation Results ---")
  for metric_name, score_value in evaluation_metrics_dict.items():
      if metric_name != "Confusion Matrix":
          print(f"{metric_name} : {score_value:.4f}")
      else:
          print(f"{metric_name} :\n{np.array(score_value)}")

  # Plot Confusion Matrix
  plt.figure(figsize=(6,4))
  sns.heatmap(evaluation_metrics_dict["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, 
              xticklabels=['Predicted: Non-Hit (0)', 'Predicted: Hit (1)'], 
              yticklabels=['Actual: Non-Hit (0)', 'Actual: Hit (1)'])
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.title("Confusion Matrix on Test Data (Logistic Regression - Best Tuned)")
  plt.show()

  fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba_test)
  roc_auc_value = auc(fpr, tpr)

  # Plot ROC Curve
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_value:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate (1 - Specificity)')
  plt.ylabel('True Positive Rate (Sensitivity/Recall)')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc="lower right")
  plt.grid(True)
  plt.show()
  print("--- Logistic Regression Experiment Complete ---")
  
  return optimized_pipeline_model, evaluation_metrics_dict, optimal_hyperparameters

# Main execution block to run the logistic regression experiment
if __name__ == "__main__":
    raw_df = pd.read_csv("/Users/orion-gz/Desktop/Project/PYTHON/rawg_games_data.csv")
    X_processed, y_target = preprocessing_rawg_data(raw_df)
    numerical_features = ['metacritic', 'playtime']
    param_grid_list = [
        { 
            'data_preprocessor__num__scaler': [StandardScaler(), RobustScaler(), MinMaxScaler(), 'passthrough'], 
            'selector__k': [50, 100, 150, X_processed.shape[1]],
            'model__solver': ['liblinear'],
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.01, 0.1, 1.0, 10.0],
            'model__class_weight': ['balanced', {0:1, 1:1.5}, {0:1, 1:2}, {0:1, 1:3}]
        }
        # Too much time consuming
        ,{ 
            'data_preprocessor__num__scaler': [StandardScaler(), RobustScaler(), MinMaxScaler(), 'passthrough'],
            'selector__k': [50, 100, 150, X_processed.shape[1]],
            'model__solver': ['lbfgs', 'saga'],
            'model__penalty': ['l2'], 
            'model__C': [0.01, 0.1, 1.0, 10.0],
            'model__class_weight': ['balanced', {0:1, 1:1.5}, {0:1, 1:2}, {0:1, 1:3}]
        }
    ]
    best_lr_model_tuned, lr_metrics_tuned, lr_params_tuned = run_logistic_regression(
        X_processed, 
        y_target,   
        numerical_features, 
        selector_instance=SelectKBest(score_func=f_classif), 
        model_params_grid=param_grid_list, 
        main_scoring_metric='f1_weighted'
    )
    print("Logistic Regression - Best Overall Tuned Parameters:", lr_params_tuned)
    print("Logistic Regression - Final Test Set Metrics (Tuned):", lr_metrics_tuned)
   
    