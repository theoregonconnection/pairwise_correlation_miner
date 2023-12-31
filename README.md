# pairwise_correlation_miner

# Pairwise Correlation Miner
_Last Update: 12/21/23_

## Summary
The `Pairwise_Correlation_Miner` package provides the `execute_feature_target_pair_analysis` function, designed to analyze statistical relationships between multiple features and targets in a dataframe, outputting the results in CSV format.

## Installation
Available as a package at PyPi.org. https://pypi.org/project/pairwise-correlation-miner/. 
Install via pip: pip install pairwise-correlation-miner

## Function Usage
`execute_feature_target_pair_analysis(dataframe, feature_list, target_list, export_file_path)`

- `dataframe`: DataFrame containing non-null, numeric feature and target values.
- `feature_list`: List of column names to be treated as features.
- `target_list`: List of column names to be treated as targets.
- `export_file_path`: File path for exporting results (e.g., `"data.csv"`).

# Function returns 
-	dataloader_df: The results of the analysis in a dataframe
-	Exports a CSV file to the path and file you define when submitting the function. 

## Example
```python
# Example usage of the function
import pairwise_correlation_miner as pcm

dataframe = ...
feature_list = ['feature1', 'feature2']
target_list = ['target1', 'target2']
export_file_path = "results.csv"

pcm.execute_feature_target_pair_analysis(dataframe, feature_list, target_list, export_file_path)
```

## Analysis Overview

The function loops through all of the possible feature-target pairs and performs 3 analyses:
1.	Linear Regression: Uses scikit-learn's linear regression, with an m-test for p-value calculation.
2.	Polynomial Regression: 2nd degree polynomial regression, also with m-test based p-value calculation.
3.	Autobucketed Welches t Test: Buckets data points for binary and non-binary targets to perform a t-test.
The m-test is a way to generate a p value for the probability that a given feature predicts a target, baed on a regression equation. For each data point, we calculate an estimate of the probability that the datapoint is as close to the regression line as it actually is, and the probability that it is as close to the null hypothesis line as it actually is. Once we have these 2 probability values for each datapoint, we run a t test to compare whether the prediction probabilities are different than the null probabilities. The null hypothesis line is set at y = (target mean + target median) / 2. 

## License 
This project is licensed under the MIT License.

