import pandas as pd
from aif360.datasets import CompasDataset

# Load the dataset
compas_data = CompasDataset()

# Convert dataset to pandas DataFrame for easier analysis
df = compas_data.convert_to_dataframe()[0]

# Mapping protected attributes
sex_map = compas_data.metadata['protected_attribute_maps'][0]
race_map = compas_data.metadata['protected_attribute_maps'][1]
label_map = compas_data.metadata['label_maps'][0]

# Add readable labels for protected attributes and labels
df['sex'] = df['sex'].map(sex_map)
df['race'] = df['race'].map(race_map)
df['two_year_recid'] = df['two_year_recid'].map(label_map)

# 1. Distribution of subpopulations based on protected attributes
subpopulation_counts = df.groupby(['sex', 'race']).size().unstack(fill_value=0)

# 2. Outcome distribution across groups
outcome_distribution = df.groupby(['sex', 'race', 'two_year_recid']).size().unstack(fill_value=0)

# Display the results
print(subpopulation_counts)
print(outcome_distribution)
