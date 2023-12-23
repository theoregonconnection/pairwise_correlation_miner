#!/usr/bin/env python
# coding: utf-8

# Import packages
import pandas as pd
from datetime import datetime
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import PolynomialFeatures 

''' EXAMPLE DATA 
# Define the Dataframe
df_to_use = df_base

# Define the target variables to analyze 
target_list = ['RESULT_HOME_TEAM_VICTORY', 
'RESULT_HOME_TEAM_FAVORITE', 
'RESULT_FAVORITE_VICTORY', 
'RESULT_SPREAD_OVER', 
'RESULT_SPREAD_UNDER' 
]

# Define the feature variables to analyze 
feature_list = ['DIV_GAME', 
'TEMPERATURE', 
'WINDSPEED', 
'GRASS', 
'WEATHER_RAIN', 
'WEATHER_SNOW', 
'WEATHER_CLEAR', 
'WEATHER_WIND_RANK', 
'WEATHER_TEMPERATURE_RANK', 
'TREND_DIFF_HOME_OV_VS_AWAY_OV_PLAY_RANK' 
]

# Specify the file path where you want to save the CSV file
export_file_path_string = r"C:\Users\theor\OneDrive\Desktop\ML_Miner\data.csv"
'''


# This sub function is used by the main function to check that the data submitted to the function has proper format
def valid_data_checker(feature_df, target_df):

    #Assess whether data has non-numeric values or nulls.
    valid_data_check = 1 

    # PROCESS FEATURE_DF
    # Check for non-numeric values
    non_numeric_columns = feature_df.select_dtypes(exclude=['number']).columns
    if not non_numeric_columns.empty:
        print(f"The feature_df DataFrame contains non-numeric values in the following columns: {non_numeric_columns.tolist()}")
        valid_data_check = 0
    # Check for null values
    null_values = feature_df.isnull()
    if null_values.any().any():
        print(f"There are {null_values.sum()} null values in the feature DataFrame for column: {feature_df.columns[0]}.")
        valid_data_check = 0

    # PROCESS TARGET_DF
    # Check for non-numeric values
    non_numeric_columns = target_df.select_dtypes(exclude=['number']).columns
    if not non_numeric_columns.empty:
        print(f"The target_df DataFrame contains non-numeric values in the following columns: {non_numeric_columns.tolist()}")
        valid_data_check = 0
    # Check for null values
    null_values = target_df.isnull()
    if null_values.any().any():
        print(f"There are {null_values.sum()} null values in the target DataFrame for column: {target_df.columns[0]}.")
        valid_data_check = 0

    # Return the continue value
    return valid_data_check


# M Test for linear regression 
# The M test takes each data point and determines 2 approximate values: the probability that the point is as close to the predicted line as it is, and the probability that the point is as close to the null hypothesis line as it is. 
# The M test then runs a t test to determine the probability that those 2 buckets of probabilities are statistically different.

def m_test_linear_regression(slope, intercept, feature_df, target_df):

    
    # Get the average, median, null default, min and max range for target_df
    target_mean = target_df.iloc[:, 0].mean()
    target_median = target_df.iloc[:, 0].median()
    null_value = ((target_mean + target_median) / 2)
    target_min = target_df.iloc[:, 0].min()
    target_max = target_df.iloc[:, 0].max()
    
    # Create placeholders for probabilities
    test_probabilities = []
    null_probabilities = []

    # Get the stats for the target_df 
    
    
    i = 0
    while i < len(feature_df):
        # Select the feature and target value to analyze
        feature_value = feature_df.iloc[i, 0]
        target_value = target_df.iloc[i, 0]

        # Predict where target would be if model was true 
        predicted_target = (slope)*(feature_value) + intercept 
        
        # Get the maximum possible distance from the predicted target value to the min/max of the possible values
        max_possible_distance_prediction = max(abs(target_min - predicted_target), abs(target_max - predicted_target))
        # Get the actual distance from the target value to the predicted target value 
        actual_distance_prediction = abs(target_value - predicted_target)
        # Get the approximate probability value for predicted vs actual 
        prediction_probability = 1 - (actual_distance_prediction/max_possible_distance_prediction)
        
        # Get the maximum possible distance from the null hypothesis value to the min/max of the possible values
        max_possible_distance_null_hypothesis = max(abs(target_min - null_value), abs(target_max - null_value))
        # Get the actual distance from the target value to the null hypothesis value 
        actual_distance_null_hypothesis = abs(target_value - null_value)
        # Get the approximate probability value for predicted vs actual 
        null_hypothesis_probability = 1 - (actual_distance_null_hypothesis/max_possible_distance_null_hypothesis)
 
        # Append probabilities
        test_probabilities.append(prediction_probability)
        null_probabilities.append(null_hypothesis_probability)        
        
        i += 1

    # Get Results 
    avg_test_probability = sum(test_probabilities)/len(test_probabilities) 
    avg_null_probability = sum(null_probabilities)/len(null_probabilities)
    if avg_test_probability > avg_null_probability:
        test_probability_higher = 1
    else:
        test_probability_higher = 0

    # Perform t-test
    t_statistic, p_value = ttest_ind(test_probabilities, null_probabilities)

    # Put results in dataframe for return
    results_data = {'AVG_TEST_PROBABILITY': [avg_test_probability], 
                   'AVG_NULL_PROBABILITY': [avg_null_probability], 
                   'TEST_PROBABILITY_HIGHER': [test_probability_higher], 
                   'T_STATISTIC': [t_statistic], 
                   'P_VALUE': [p_value], 
                    'DATA_POINTS': [len(feature_df)]
                  }

    results_df = pd.DataFrame(results_data)

    return results_df 


#Run the function 
#m_test_linear_regression(slope, intercept, f_df, t_df) 



# Function to execute linear regression analysis 

def execute_linear_regression(feature_df, target_df):

    # Concatenate the two dataframes horizontally
    data = pd.concat([feature_df, target_df], axis=1)
    feature_name = feature_df.columns[0]
    target_name = target_df.columns[0]

    # Create a pandas series for the target variable 
    target_series = target_df[target_name]
    
    #Create a dataframe to hold the results of the loop
    # Create a DataFrame with three columns
    results_data = {
        'slope': [],
        'intercept': [],
        'mse': [],
        'r2': [], 
        'AVG_TEST_PROBABILITY': [], 
        'AVG_NULL_PROBABILITY': [],
        'TEST_PROBABILITY_HIGHER': [],
        'T_STATISTIC': [],
        'M_TEST_P_VALUE': [],
        'DATA_POINTS': []
    }

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model on the data
    model.fit(feature_df, target_series)

    # Get the stats for linear regression
    predictions = model.predict(feature_df)
    mse = mean_squared_error(target_series, predictions)
    r2 = r2_score(target_series, predictions)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Get the probability stats from the m test for linear regression
    probability_stats  =  m_test_linear_regression(slope, intercept, feature_df, target_df) 
    
    # Load results to the results_data dictionary
    results_data['slope'].append(slope)
    results_data['intercept'].append(intercept)
    results_data['mse'].append(mse)
    results_data['r2'].append(r2)
    results_data['AVG_TEST_PROBABILITY'].append(probability_stats.loc[0, 'AVG_TEST_PROBABILITY'])
    results_data['AVG_NULL_PROBABILITY'].append(probability_stats.loc[0, 'AVG_NULL_PROBABILITY'])
    results_data['TEST_PROBABILITY_HIGHER'].append(probability_stats.loc[0, 'TEST_PROBABILITY_HIGHER'])
    results_data['T_STATISTIC'].append(probability_stats.loc[0, 'T_STATISTIC'])
    results_data['M_TEST_P_VALUE'].append(probability_stats.loc[0, 'P_VALUE'])
    results_data['DATA_POINTS'].append(probability_stats.loc[0, 'DATA_POINTS'])
    
    return results_data


#execute_linear_regression(feature_df, target_df)
#print(feature_df)
#print(target_df)



# M Test for 2 degree polynomial regression 
# The M test takes each data point and determines 2 approximate values: the probability that the point is as close to the predicted line as it is, and the probability that the point is as close to the null hypothesis line as it is. 
# The M test then runs a t test to determine the probability that those 2 buckets of probabilities are statistically different.

def m_test_polynomial_2_regression(coefficient1, coefficient2, intercept, feature_df, target_df):
    
    # Get the average, median, null default, min and max range for target_df
    target_mean = target_df.iloc[:, 0].mean()
    target_median = target_df.iloc[:, 0].median()
    null_value = ((target_mean + target_median) / 2)
    target_min = target_df.iloc[:, 0].min()
    target_max = target_df.iloc[:, 0].max()
    
    # Create placeholders for probabilities
    test_probabilities = []
    null_probabilities = []

    # Get the stats for the target_df    
    i = 0
    while i < len(feature_df):
        # Select the feature and target value to analyze
        feature_value = feature_df.iloc[i, 0]
        target_value = target_df.iloc[i, 0]

        # Predict where target would be if model was true 
        predicted_target = ((coefficient1)*(feature_value)) + ((coefficient2)*((feature_value)**2)) + intercept 
  
        # Get the maximum possible distance from the predicted target value to the min/max of the possible values
        max_possible_distance_prediction = max(abs(target_min - predicted_target), abs(target_max - predicted_target))
        # Get the actual distance from the target value to the predicted target value 
        actual_distance_prediction = abs(target_value - predicted_target)
        # Get the approximate probability value for predicted vs actual 
        prediction_probability = 1 - (actual_distance_prediction/max_possible_distance_prediction)
        
        # Get the maximum possible distance from the null hypothesis value to the min/max of the possible values
        max_possible_distance_null_hypothesis = max(abs(target_min - null_value), abs(target_max - null_value))
        # Get the actual distance from the target value to the null hypothesis value 
        actual_distance_null_hypothesis = abs(target_value - null_value)
        # Get the approximate probability value for predicted vs actual 
        null_hypothesis_probability = 1 - (actual_distance_null_hypothesis/max_possible_distance_null_hypothesis)
 
        # Append probabilities
        test_probabilities.append(prediction_probability)
        null_probabilities.append(null_hypothesis_probability)        
        
        
        #print(f"Predicted Target: {predicted_target}")
        #print(f"Actual Target: {target_value}")
        
        i += 1
    
    # Get Results 
    avg_test_probability = sum(test_probabilities)/len(test_probabilities) 
    avg_null_probability = sum(null_probabilities)/len(null_probabilities)
    if avg_test_probability > avg_null_probability:
        test_probability_higher = 1
    else:
        test_probability_higher = 0

    # Perform t-test
    t_statistic, p_value = ttest_ind(test_probabilities, null_probabilities)

    # Put results in dataframe for return
    results_data = {'AVG_TEST_PROBABILITY': [avg_test_probability], 
                   'AVG_NULL_PROBABILITY': [avg_null_probability], 
                   'TEST_PROBABILITY_HIGHER': [test_probability_higher], 
                   'T_STATISTIC': [t_statistic], 
                   'P_VALUE': [p_value], 
                   'DATA_POINTS': [len(feature_df)]
                  }

    results_df = pd.DataFrame(results_data)

    return results_df 

#Run the function 
#output = m_test_polynomial_2_regression(coefficient1, coefficient2, intercept, feature_df, target_df) 
#print(output)


# Function to execute 2 degree polynomial regression analysis 

def execute_polynomial_regression(feature_df, target_df):
    
    # Concatenate the two dataframes horizontally
    data = pd.concat([feature_df, target_df], axis=1)
    feature_name = feature_df.columns[0]
    target_name = target_df.columns[0]
    
    # Create a pandas series for the target variable 
    target_series = target_df[target_name]
    
    #Create a dataframe to hold the results of the loop
    # Create a DataFrame with three columns
    results_data = {
    'coefficient1': [],
    'coefficient2': [],
    'intercept': [],
    'mse': [],
    'r2': [], 
    'AVG_TEST_PROBABILITY': [], 
    'AVG_NULL_PROBABILITY': [],
    'TEST_PROBABILITY_HIGHER': [],
    'T_STATISTIC': [],
    'M_TEST_P_VALUE': [],
    'DATA_POINTS': []
    }
    
    
    # Define the degree of the polynomial
    degree = 2  # You can change this as needed
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(feature_df)

    # Fit a linear regression model to the polynomial features
    model = LinearRegression()
    model.fit(X_poly, target_series)

    # Make predictions on the entire dataset
    y_pred = model.predict(X_poly)
    
    # Print the coefficients of the polynomial regression equation
    coefficients = model.coef_
    coefficient1 = coefficients[1]
    coefficient2 = coefficients[2]
    intercept = model.intercept_

    
    # Evaluate the model
    mse = mean_squared_error(target_series, y_pred)
    r2 = r2_score(target_series, y_pred)

    # Get the probability stats from the m test for linear regression
    probability_stats  =  m_test_polynomial_2_regression(coefficient1, coefficient2, intercept, feature_df, target_df) 
    
    # Load results to the results_data dictionary
    results_data['coefficient1'].append(coefficient1)
    results_data['coefficient2'].append(coefficient2)
    results_data['intercept'].append(intercept)
    results_data['mse'].append(mse)
    results_data['r2'].append(r2)
    results_data['AVG_TEST_PROBABILITY'].append(probability_stats.loc[0, 'AVG_TEST_PROBABILITY'])
    results_data['AVG_NULL_PROBABILITY'].append(probability_stats.loc[0, 'AVG_NULL_PROBABILITY'])
    results_data['TEST_PROBABILITY_HIGHER'].append(probability_stats.loc[0, 'TEST_PROBABILITY_HIGHER'])
    results_data['T_STATISTIC'].append(probability_stats.loc[0, 'T_STATISTIC'])
    results_data['M_TEST_P_VALUE'].append(probability_stats.loc[0, 'P_VALUE'])
    results_data['DATA_POINTS'].append(probability_stats.loc[0, 'DATA_POINTS'])
    
    return results_data

#execute_polynomial_regression(feature_df, target_df)
#print(feature_df)
#print(target_df)


# Autobucket and Execute T Test 
# The autobucket T test automatically separates the target data into binary low vs high buckets. The threshold is mean(mean, median). Then it runs a Welches t test to asses whether the feature values from the low bucket are different than the feature values from the high bucket. 
# Note: If your data is already in binary 0/1 format, this should keep them in the same binary buckets (unless you only have 1 represented, of course). 

def autobucket_t_test(feature_df, target_df):

     # Get the average, median and blended average of target
    target_mean = target_df.iloc[:, 0].mean()
    target_median = target_df.iloc[:, 0].median()
    blended_average = ((target_mean + target_median) / 2)
    

    # Put feature variables into high vs low. 
    features_with_low_target = []
    features_with_high_target = []
    
    i = 0
    while i < len(feature_df):
        # Select the feature and target value to analyze
        feature_value = feature_df.iloc[i, 0]
        target_value = target_df.iloc[i, 0]

        if target_value >= blended_average:
            features_with_high_target.append(feature_value) 
        else:
            features_with_low_target.append(feature_value)
        i += 1

    
    number_of_high_target_features = len(features_with_high_target)
    number_of_low_target_features = len(features_with_low_target)
    avg_feature_value_high_target = sum(features_with_high_target)/len(features_with_high_target)
    avg_feature_value_low_target = sum(features_with_low_target)/len(features_with_low_target)

    # Perform Welch's t-test
    t_statistic, p_value = ttest_ind(features_with_high_target, features_with_low_target, equal_var=False)

    results_data = {
    'Threshold': [blended_average],
    'High_Target_Feature_Count': [number_of_high_target_features],
    'Low_Target_Feature_Count': [number_of_low_target_features],
    'High_Target_Feature_Avg': [avg_feature_value_high_target],
    'Low_Target_Feature_Avg': [avg_feature_value_low_target], 
    'T_Statistic': [t_statistic], 
    'p_Value': [p_value]
    }

    return results_data

#autobucket_t_test(feature_df, target_df)



# This is the main function. You feed it a dataframe with all of your data, a list of the columns you want to treat as features, a list of columns to treat as targets and the location where you want to export the results to. 

def execute_feature_target_pair_analysis(df_to_use, feature_list, target_list, export_file_path_string): 
    
    try:
        # Validate that inputs are dataframe, list, list. Exit the function if they are not valid. 
        proceed = 1
        if not isinstance(df_to_use, pd.DataFrame):
            print(f"FAILURE: The first object (dataframe with all data) submitted to the function must be a dataframe. You are currently submitting a {type(target_list)}.")
            proceed = 0
        if not isinstance(feature_list, list):
            print(f"FAILURE: The second object (feature_list) submitted to the function must be a list that has the column names from the dataframe that you want to define as feature variables. You are currently submitting a {type(feature_list)}.")
            proceed = 0
        if not isinstance(target_list, list):
            print(f"FAILURE: The third object (target_list) submitted to the function must be a list that has the column names from the dataframe that you want to define as target variables. You are currently submitting a {type(target_list)}.")
            proceed = 0
        if proceed == 0:
            print("Exiting function because of improper input data")
            return
        
        
        # Begin processing data 
        # Instantiate dataloader dataframe
        dataloader_df = pd.DataFrame(columns=[
        'feature', 'target', 'data_points', 'avg_p_value', 
        'lr_slope', 'lr_intercept', 'lr_mse', 'lr_r2', 'lr_avg_test_probability', 'lr_avg_null_probability', 'lr_t_statistic', 'lr_p_value', 
        'pr_coefficient_1', 'pr_coefficient_2', 'pr_intercept', 'pr_mse', 'pr_r2', 'pr_avg_test_probability', 'pr_avg_null_probability', 'pr_t_statistic', 'pr_p_value', 
        'att_threshold', 'att_high_target_feature_count', 'att_low_target_feature_count', 'att_high_target_feature_avg', 'att_low_target_feature_avg', 'att_t_statistic', 'att_p_value'
        ])
        
        # Calculate # of values
        print(f"Processing {len(feature_list)} feature values and {len(target_list)} target values for a total of {len(feature_list)*len(target_list)} combinations.")
        
        # Initiate Loop
        target_count = 0
        for target in target_list:
            # Report progress
            target_count += 1
            print(f"Currently processing target {target_count} of {len(target_list)} total targets.", end='\r')
            
            # Create target_df
            target_df = df_to_use[[target]].copy()
            
            for feature in feature_list:
                feature_df = df_to_use[[feature]].copy()
                
                # Check whether data is valid
                valid_data = valid_data_checker(feature_df, target_df)
                
                if valid_data == 1:
                    # Execute models if data is valid
                    #print(f"Processing Data for Feature = {feature_df.columns[0]} and Target = {target_df.columns[0]} ")
                    linear_regression_results = execute_linear_regression(feature_df, target_df)
                    polynomial_regression_results = execute_polynomial_regression(feature_df, target_df)
                    autobucket_t_test_results = autobucket_t_test(feature_df, target_df)
                
                    # Get the p value average for the 3 tests
                    avg_p_value = (linear_regression_results['M_TEST_P_VALUE'][0] + polynomial_regression_results['M_TEST_P_VALUE'][0] + autobucket_t_test_results['p_Value'][0]) / 3
                
                    # Create Data Loader 
                    dataloader = {
                        'feature': feature_df.columns[0],
                        'target': target_df.columns[0],
                        'data_points': len(feature_df),
                        'avg_p_value': avg_p_value, 
                        'lr_slope': linear_regression_results['slope'][0], 
                        'lr_intercept': linear_regression_results['intercept'][0], 
                        'lr_mse': linear_regression_results['mse'][0], 
                        'lr_r2': linear_regression_results['r2'][0], 
                        'lr_avg_test_probability': linear_regression_results['AVG_TEST_PROBABILITY'][0], 
                        'lr_avg_null_probability': linear_regression_results['AVG_NULL_PROBABILITY'][0], 
                        'lr_t_statistic': linear_regression_results['T_STATISTIC'][0], 
                        'lr_p_value': linear_regression_results['M_TEST_P_VALUE'][0], 
                
                        'pr_coefficient_1': polynomial_regression_results['coefficient1'][0], 
                        'pr_coefficient_2': polynomial_regression_results['coefficient2'][0], 
                        'pr_intercept': polynomial_regression_results['intercept'][0], 
                        'pr_mse': polynomial_regression_results['mse'][0], 
                        'pr_r2': polynomial_regression_results['r2'][0], 
                        'pr_avg_test_probability': polynomial_regression_results['AVG_TEST_PROBABILITY'][0], 
                        'pr_avg_null_probability': polynomial_regression_results['AVG_NULL_PROBABILITY'][0], 
                        'pr_t_statistic': polynomial_regression_results['T_STATISTIC'][0], 
                        'pr_p_value': polynomial_regression_results['M_TEST_P_VALUE'][0], 
                
                        'att_threshold': autobucket_t_test_results['Threshold'][0], 
                        'att_high_target_feature_count': autobucket_t_test_results['High_Target_Feature_Count'][0], 
                        'att_low_target_feature_count': autobucket_t_test_results['Low_Target_Feature_Count'][0], 
                        'att_high_target_feature_avg': autobucket_t_test_results['High_Target_Feature_Avg'][0], 
                        'att_low_target_feature_avg': autobucket_t_test_results['Low_Target_Feature_Avg'][0], 
                        'att_t_statistic': autobucket_t_test_results['T_Statistic'][0], 
                        'att_p_value': autobucket_t_test_results['p_Value'][0]
                    }
                
                    # Append the new data to the DataFrame
                    dataloader_df = pd.concat([dataloader_df, pd.DataFrame([dataloader])], ignore_index=True, sort=False)
                
                    #print(f"Dataloader df: {dataloader}; datatype = {type(dataloader)}")
                else:
                    print(f"Skipping Data for Feature = {feature_df.columns[0]} and Target = {target_df.columns[0]} b/c of invalid data")
                
                # Export Data 
                # Export the DataFrame to CSV
        dataloader_df.to_csv(export_file_path_string, index=False)

        # Return the results in the dataloader_df
        return dataloader_df
        print(f"Processing complete and results exported to: {export_file_path_string}")
    
    except ExceptionType as e:
            # Code to handle the exception
            print(f"An exception of type {type(e).__name__} occurred when running execute_feature_target_pair_analysis() function: {str(e)}")


# Execute the function 
execute_feature_target_pair_analysis(df_to_use, feature_list, target_list, export_file_path_string) 
