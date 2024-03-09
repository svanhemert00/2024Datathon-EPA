# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# streamlit configurations
st.set_page_config(layout="wide")

# title
st.markdown("<h1 style='text-align: center;'>🍃</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>U.S. ENVIRONMENTAL PROTECTION AGENCY (EPA)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This predictor tool is designed for women-owned businesses looking to understand expected funding of an EPA sustainability grant</p>", unsafe_allow_html=True)

# ingest data
@st.cache_resource
def load_data(file_path):
    return pd.read_csv(file_path)

data_2022 = load_data("FY2022_068_Contracts_Full_20240214_1.csv")
data_2023 = load_data("FY2023_068_Contracts_Full_20240214_1.csv")
data_2024 = load_data("FY2024_068_Contracts_Full_20240214_1.csv")
df_combined = pd.concat([data_2022, data_2023, data_2024], ignore_index=True)

# retain desired fields
field_names = [    
    "potential_total_value_of_award",
    "recipient_state_code",
    "alaskan_native_corporation_owned_firm",
    "native_hawaiian_organization_owned_firm",
    "subcontinent_asian_asian_indian_american_owned_business",
    "asian_pacific_american_owned_business",
    "black_american_owned_business",
    "hispanic_american_owned_business",
    "native_american_owned_business",
    "other_minority_owned_business",
    "veteran_owned_business",
    "woman_owned_business",
    "contracting_officers_determination_of_business_size"
]
df_combined = df_combined[field_names]

# drop duplicates
df_combined.drop_duplicates(inplace=True)

# remove outliers
Q1 = df_combined['potential_total_value_of_award'].quantile(0.25)
Q3 = df_combined['potential_total_value_of_award'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_filtered = df_combined[(df_combined['potential_total_value_of_award'] >= lower_bound) &(df_combined['potential_total_value_of_award'] <= upper_bound)]

# drop nulls and zeros
df_filtered.dropna(inplace=True)
df_filtered = df_filtered[df_filtered['potential_total_value_of_award'] != 0]

# rename columns for clarity
df_filtered.rename(columns={
    'recipient_state_code': 'region_code',
    'alaskan_native_corporation_owned_firm': 'alaskan_native_owned',
    'native_hawaiian_organization_owned_firm': 'hawaiian_native_owned',
    'subcontinent_asian_asian_indian_american_owned_business': 'asian_owned',
    'asian_pacific_american_owned_business': 'asian_pacific_owned',
    'black_american_owned_business': 'black_owned',
    'hispanic_american_owned_business': 'hispanic_owned',
    'native_american_owned_business': 'native_american_owned',
    'other_minority_owned_business': 'other_minority_owned',
    'contracting_officers_determination_of_business_size': 'business_size',
    'veteran_owned_business': 'veteran_owned',
    'woman_owned_business': 'women_owned'
}, inplace=True)

# replace t's and f's for 1's and 0's
df_filtered.replace({'t': 1, 'f': 0}, inplace=True)

# create ethnicity_sum column
ethnicity_cols = ['alaskan_native_owned', 'hawaiian_native_owned', 'asian_owned', 'asian_pacific_owned', 'black_owned', 'hispanic_owned', 'native_american_owned', 'other_minority_owned']
df_filtered['ethnicity_sum'] = df_filtered[ethnicity_cols].sum(axis=1)
df_filtered = df_filtered[df_filtered['ethnicity_sum'] != 2]
df_filtered['ethnicity_sum'].replace({0: 1}, inplace=True)

# one hot encoding and column renaming
df_filtered = pd.get_dummies(df_filtered, columns=['business_size'], drop_first=True)
df_filtered.rename(columns={'ethnicity_sum': 'white_owned'}, inplace=True)

# map state code to larger-encompassing region already used by EPA
region_mapping = {
    'CT': 1, 'ME': 1, 'MA': 1, 'NH': 1, 'RI': 1, 'VT': 1,
    'NJ': 2, 'NY': 2, 'PR': 2,
    'DE': 3, 'DC': 3, 'MD': 3, 'PA': 3, 'VA': 3, 'WV': 3,
    'AL': 4, 'FL': 4, 'GA': 4, 'KY': 4, 'MS': 4, 'NC': 4, 'SC': 4, 'TN': 4,
    'IL': 5, 'IN': 5, 'MI': 5, 'MN': 5, 'OH': 5, 'WI': 5,
    'AR': 6, 'LA': 6, 'NM': 6, 'OK': 6, 'TX': 6,
    'IA': 7, 'KS': 7, 'MO': 7, 'NE': 7,
    'CO': 8, 'MT': 8, 'ND': 8, 'SD': 8, 'UT': 8, 'WY': 8,
    'AZ': 9, 'CA': 9, 'HI': 9, 'NV': 9,
    'AK': 10, 'ID': 10, 'OR': 10, 'WA': 10
}
df_filtered['region_code'] = df_filtered['region_code'].map(region_mapping)
df_filtered.dropna(inplace=True)
df_filtered['region_code'] = df_filtered['region_code'].astype(int)

# map original ethnicity value to new ethnicity column
for index, row in df_filtered.iterrows():
    if row['alaskan_native_owned'] == 1:
        df_filtered.at[index, 'ethnicity'] = 'alaskan_native_owned'
    elif row['hawaiian_native_owned'] == 1:
        df_filtered.at[index, 'ethnicity'] = 'hawaiian_native_owned'
    elif row['asian_owned'] == 1:
        df_filtered.at[index, 'ethnicity'] = 'asian_owned'
    elif row['asian_pacific_owned'] == 1:
        df_filtered.at[index, 'ethnicity'] = 'asian_pacific_owned'
    elif row['black_owned'] == 1:
        df_filtered.at[index, 'ethnicity'] = 'black_owned'
    elif row['hispanic_owned'] == 1:
        df_filtered.at[index, 'ethnicity'] = 'hispanic_owned'
    elif row['native_american_owned'] == 1:
        df_filtered.at[index, 'ethnicity'] = 'native_american_owned'
    elif row['other_minority_owned'] == 1:
        df_filtered.at[index, 'ethnicity'] = 'other_minority_owned'
    else:
        df_filtered.at[index, 'ethnicity'] = 'white_owned'

# drop original ethnicity columns
df_filtered.drop(ethnicity_cols, axis=1, inplace=True)
df_filtered.drop('white_owned', axis=1, inplace=True)

# rename column for clarity
df_filtered.rename(columns={'business_size_SMALL BUSINESS': 'small_business'}, inplace=True)

# remove rows where ethnicity is white, retain rows where business is women-owned and drop women-owned column
df_filtered = df_filtered[df_filtered['ethnicity'] != 'white_owned']
df_filtered = df_filtered[df_filtered['women_owned'] == 1]
df_filtered.drop(['women_owned'], axis=1, inplace=True)

# streamlit user input
df_filtered['training_or_user_input'] = 'training'
potential_award_median = df_filtered['potential_total_value_of_award'].median()
states_and_territories = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PR', 'DE', 'DC' , 'MD', 'PA', 'VA', 'WV', 'AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN', 'IL', 'IN', 'MI', 'MN', 'OH', 'WI', 'AR', 'LA', 'NM', 'OK', 'TX', 'IA', 'KS', 'MO', 'NE', 'CO', 'MT', 'ND', 'SD', 'UT', 'WY', 'AZ', 'CA', 'HI', 'NV', 'AK', 'ID', 'OR', 'WA']
sorted_states_and_territories = sorted(states_and_territories)
user_state = st.selectbox('STATE CODE', sorted_states_and_territories)
ethnicities = ['ALASKAN NATIVE', 'HAWAIIAN NATIVE', 'ASIAN', 'PACIFIC ISLANDER', 'BLACK', 'HISPANIC', 'NATIVE AMERICAN', 'OTHER MINORITY']
sorted_ethnicity = sorted(ethnicities)
user_ethnicity = st.selectbox('ETHNICITY (MINORITY)', ['ALASKAN NATIVE', 'HAWAIIAN NATIVE', 'ASIAN', 'PACIFIC ISLANDER', 'BLACK', 'HISPANIC', 'NATIVE AMERICAN', 'OTHER MINORITY'])
user_small_business = st.selectbox('SMALL BUSINESS', ['YES', 'NO'])
user_veteran_status = st.selectbox('VETERAN STATUS', ['VETERAN', 'NOT A VETERAN'])
user_input = [potential_award_median, user_state, user_veteran_status, user_small_business, user_ethnicity, 'user_input']
df_filtered = df_filtered.append(pd.Series(user_input, index=df_filtered.columns), ignore_index=True)

# label encode ethnicity column
label_encoder = LabelEncoder()
df_filtered['ethnicity'] = label_encoder.fit_transform(df_filtered['ethnicity'])

# target-mean and target-median encoding
target_median_encoding_region_code = df_filtered.groupby('region_code')['potential_total_value_of_award'].median()
target_mean_encoding_veteran_owned = df_filtered.groupby('veteran_owned')['potential_total_value_of_award'].mean()
target_mean_encoding_small_business = df_filtered.groupby('small_business')['potential_total_value_of_award'].mean()
target_median_encoding_ethnicity = df_filtered.groupby('ethnicity')['potential_total_value_of_award'].median()
df_filtered['region_code'] = df_filtered['region_code'].map(target_median_encoding_region_code)
df_filtered['veteran_owned'] = df_filtered['veteran_owned'].map(target_mean_encoding_veteran_owned)
df_filtered['small_business'] = df_filtered['small_business'].map(target_mean_encoding_small_business)
df_filtered['ethnicity'] = df_filtered['ethnicity'].map(target_median_encoding_ethnicity)

# standardization
scaler = StandardScaler()
fields_to_scale = ['region_code', 'small_business', 'veteran_owned', 'ethnicity']
df_filtered[fields_to_scale] = scaler.fit_transform(df_filtered[fields_to_scale])

# bin target feature to discrete value based on tertiles for multiclass model
df_filtered_class = df_filtered.copy()
df_filtered_class['potential_total_value_of_award'] = pd.qcut(df_filtered['potential_total_value_of_award'], q=3, labels=[1, 2, 3])

# identifying tertiles
t1 = np.percentile(df_filtered['potential_total_value_of_award'], 33.33)
t2 = np.percentile(df_filtered['potential_total_value_of_award'], 66.66)

# partition user input from training data
user_input = df_filtered_class[df_filtered_class['training_or_user_input'] == 'user_input'][['region_code', 'veteran_owned', 'small_business', 'ethnicity']]
df_filtered_class = df_filtered_class[df_filtered_class['training_or_user_input'] != 'user_input']
df_filtered_class = df_filtered_class[['potential_total_value_of_award', 'region_code', 'veteran_owned', 'small_business', 'ethnicity']]

# define dependent and independent variables in dataframe
X = df_filtered_class.drop(columns=['potential_total_value_of_award'])
y = df_filtered_class['potential_total_value_of_award']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# apply SMOTE to handle class imbalances
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

# make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# calculate the accuracy score
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)

# type cast tertiles
t1 = str(int(t1))
t2 = str(int(t2))

# predict user input
user_pred = rf_classifier.predict(user_input)
if user_pred == 1:
    grant_range = '😔 SMALL-SIZED GRANT (\$0-\$' + t1 + ') 🤏'
elif user_pred == 2:
    grant_range = 'MEDIUM-SIZED GRANT (\$' + t1 + '-\$' + t2 + ')'
else:
    grant_range = 'LARGE-SIZED GRANT (\$'+ t2 + '+)'
          
# run button
run_button = st.button("🏃 RUN")       
if run_button:
    st.success(grant_range)
    
# display data button
#display = st.button("DISPLAY DATA")       
#if run_button:
#    st.write(df_combined)

# image logo
#st.image('./EPA_logo.png')

# links
st.link_button('🌐 EPA.GOV', "https://www.epa.gov/")
st.link_button('🐱 GITHUB REPO', "https://github.com/svanhemert00/2024Datathon-EPA")
display_data = st.checkbox("💾 SHOW DATA")     
if display_data:
    st.write(df_combined)