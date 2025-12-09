
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

st.title('House Price Prediction App')

# --- Load Models and Encoders ---
# @st.cache_resource is used to cache the model loading for performance
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model('best_gradient_boosting_regressor.joblib')
label_encoder = load_model('label_encodings.joblib')

# Re-initialize and fit OneHotEncoder with known categories
# The original OneHotEncoder was not saved in a fitted state, so we reconstruct it.

# Extract categories from model.feature_names_in_ for one-hot encoded features
model_features = model.feature_names_in_.tolist()
onehot_base_features = ['GarageFinish', 'Neighborhood']

extracted_categories = {feature: [] for feature in onehot_base_features}

for model_feature in model_features:
    for base_feature in onehot_base_features:
        if model_feature.startswith(f'{base_feature}_'):
            category = model_feature[len(f'{base_feature}_'):]
            if category: # Ensure not to add empty strings as categories
                extracted_categories[base_feature].append(category)

# Sort categories for consistency
for feature in extracted_categories:
    extracted_categories[feature].sort()

garage_finish_categories = extracted_categories['GarageFinish']
neighborhood_categories = extracted_categories['Neighborhood']

onehot_encoder = OneHotEncoder(categories=[garage_finish_categories, neighborhood_categories],
                               handle_unknown='ignore',
                               sparse_output=False)

# Create dummy data to 'fit' the onehot_encoder so it can be used to transform
dummy_fit_data = []
for gf_cat in garage_finish_categories:
    for nh_cat in neighborhood_categories:
        dummy_fit_data.append([gf_cat, nh_cat])

if dummy_fit_data:
    dummy_fit_df = pd.DataFrame(dummy_fit_data, columns=onehot_base_features)
    onehot_encoder.fit(dummy_fit_df)


st.subheader('Enter House Features:')

# --- Input Widgets for Features (hardcoded defaults from kernel state) ---
# Numerical features
first_flr_sf = st.number_input('1st Floor Square Feet', min_value=0, value=1092)
second_flr_sf = st.number_input('2nd Floor Square Feet', min_value=0, value=0)
bsmt_fin_sf1 = st.number_input('Basement Finished Square Feet 1', min_value=0, value=680)
full_bath = st.number_input('Full Bathrooms', min_value=0, max_value=4, value=1)
garage_cars = st.number_input('Garage Car Capacity', min_value=0, max_value=5, value=2)
gr_liv_area = st.number_input('Above Grade Living Area (SqFt)', min_value=0, value=1350)
lot_area = st.number_input('Lot Size (Square Feet)', min_value=0, value=11369)
lot_frontage = st.number_input('Lot Frontage (feet)', min_value=0, value=67)
mas_vnr_area = st.number_input('Masonry Veneer Area (SqFt)', min_value=0, value=204)
open_porch_sf = st.number_input('Open Porch Square Feet', min_value=0, value=31)
tot_rms_abv_grd = st.number_input('Total Rooms Above Grade', min_value=0, value=6)
total_bsmt_sf = st.number_input('Total Basement Square Feet', min_value=0, value=1016)
wood_deck_sf = st.number_input('Wood Deck Square Feet', min_value=0, value=20)
year_built = st.number_input('Year Built', min_value=1800, max_value=2025, value=1971)
year_remod_add = st.number_input('Year Remodel/Add', min_value=1800, max_value=2025, value=1974)

# Categorical features
bsmt_qual = st.selectbox('Basement Quality', options=list(label_encoder['bsmtqual_mapping'].keys()), index=0) # 'Ex'
central_air = st.selectbox('Central Air', options=list(label_encoder['centralair_mapping'].keys()), index=0) # 'Y'
exter_qual = st.selectbox('Exterior Quality', options=list(label_encoder['exterqual_mapping'].keys()), index=0) # 'Ex'
kitchen_qual = st.selectbox('Kitchen Quality', options=list(label_encoder['kitchenqual_mapping'].keys()), index=0) # 'Ex'
overall_qual = st.selectbox('Overall Quality', options=[1, 2, 3, 4, 5, 6, 7, 8, 9], index=6) # 7
garage_finish = st.selectbox('Garage Finish', options=garage_finish_categories, index=0) # 'Fin'
neighborhood = st.selectbox('Neighborhood', options=neighborhood_categories, index=0) # 'Blmngtn'

# --- Prediction Logic ---
if st.button('Predict Sale Price'):
    user_input_dict = {
        '1stFlrSF': first_flr_sf,
        '2ndFlrSF': second_flr_sf,
        'BsmtFinSF1': bsmt_fin_sf1,
        'BsmtQual': bsmt_qual,
        'CentralAir': central_air,
        'ExterQual': exter_qual,
        'FullBath': full_bath,
        'GarageCars': garage_cars,
        'GrLivArea': gr_liv_area,
        'KitchenQual': kitchen_qual,
        'LotArea': lot_area,
        'LotFrontage': lot_frontage,
        'MasVnrArea': mas_vnr_area,
        'OpenPorchSF': open_porch_sf,
        'OverallQual': overall_qual,
        'TotRmsAbvGrd': tot_rms_abv_grd,
        'TotalBsmtSF': total_bsmt_sf,
        'WoodDeckSF': wood_deck_sf,
        'YearBuilt': year_built,
        'YearRemodAdd': year_remod_add,
        'GarageFinish': garage_finish,
        'Neighborhood': neighborhood
    }

    input_df = pd.DataFrame([user_input_dict])

    # One-hot encode features
    input_onehot_df = pd.DataFrame(onehot_encoder.transform(input_df[onehot_base_features]),
                                   columns=onehot_encoder.get_feature_names_out(onehot_base_features),
                                   index=input_df.index)
    input_df = input_df.drop(columns=onehot_base_features)
    input_df = pd.concat([input_df, input_onehot_df], axis=1)

    # Apply label encoding
    for col, mapping_key in [
        ('BsmtQual', 'bsmtqual_mapping'),
        ('CentralAir', 'centralair_mapping'),
        ('ExterQual', 'exterqual_mapping'),
        ('KitchenQual', 'kitchenqual_mapping')
    ]:
        if col in input_df.columns:
            input_df[col] = input_df[col].map(label_encoder[mapping_key]).astype('int64')

    # Convert OverallQual to int64
    if 'OverallQual' in input_df.columns:
        input_df['OverallQual'] = input_df['OverallQual'].astype('int64')

    # Align columns with model's expected features
    missing_cols = set(model_features) - set(input_df.columns)
    for c in missing_cols:
        input_df[c] = 0

    extra_cols_in_input = set(input_df.columns) - set(model_features)
    if extra_cols_in_input:
        input_df = input_df.drop(columns=list(extra_cols_in_input))

    input_df = input_df[model_features]

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display prediction
    st.success(f'Predicted Sale Price: ${prediction:,.2f}')
