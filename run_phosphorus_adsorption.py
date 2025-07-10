import joblib
import pandas as pd

url = "https://zenodo.org/record/15854383/files/MEGADATA.xlsx"
df = pd.read_excel(url)

pd.options.mode.chained_assignment = None  # default='warn'

def classify_soil(texture):
    clayey = ["C", "CL", "SiC", "SiCL", "SC"]
    loamy  = ["L", "SiL", "SCL"]
    sandy  = ["S", "LS", "SL"]

    if texture in clayey:
        return "Clayey"
    elif texture in loamy:
        return "Loamy"
    elif texture in sandy:
        return "Sandy"
    else:
        return "Unknown"

# Apply the function to create a new column
df['Soil Classification'] = df['Soil texture'].apply(classify_soil)

df.columns = df.columns.str.strip()


required_columns = [
    'Soil texture', 'S', 'C', 'Si', 'pH', 'Ec', 'O.O.', 'CaCO3',
    'NO3', 'NO3-N', 'P', 'K', 'Mg', 'Ca', 'Fe', 'Zn', 'Mn', 'Cu', 'B'
]

# Drop rows with NaNs in any of the required columns
df_cleaned = df.dropna(subset=required_columns)

for col in df_cleaned.select_dtypes(include='object').columns:
    df_cleaned[col] = df_cleaned[col].astype(str).str.replace(',', '.', regex=False)
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')  # Keeps invalid values as strings

target_columns = ['S', 'C', 'Si', 'pH', 'Ec', 'O.O.', 'CaCO3',
                  'NO3-N', 'P', 'K', 'Mg', 'Fe', 'Zn', 'Mn', 'Cu', 'B']

# Convert strings to NaN, and keep numeric values
for col in target_columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

target_columns = ['S', 'C', 'Si', 'pH', 'Ec', 'O.O.', 'CaCO3',
                  'NO3-N', 'P', 'K', 'Mg', 'Fe', 'Zn', 'Mn', 'Cu', 'B']

# Drop rows with NaNs in those columns
df_cleaned = df_cleaned.dropna(subset=target_columns)

df_expanded = df_cleaned.copy()

df_expanded.rename(columns={'O.O.': 'Organic matter', 'Ec': 'EC'}, inplace=True)

df_expanded['Mg'] = df_expanded['Mg'] / (12.1525 * 10)

X_input = df_expanded[['S', 'C', 'pH', 'EC', 'Organic matter', 'P', 'Mg', 'Mn', 'Cu']]

# Load the model
model_multi = joblib.load('multioutput_xgb_model.pkl')

# predict phosphorus adsorption for the extended dataset
y_pred = model_multi.predict(X_input)

# 1. Convert predictions to DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=['XGBoost_1ppm_adsorbed', 'XGBoost_2ppm_adsorbed', 'XGBoost_4ppm_adsorbed',
                                         'XGBoost_6ppm_adsorbed', 'XGBoost_10ppm_adsorbed'])

# 2. Reset index (optional, if needed for alignment)
y_pred_df.reset_index(drop=True, inplace=True)
df_expanded.reset_index(drop=True, inplace=True)

# 3. Concatenate to df_final
df_expanded = pd.concat([df_expanded, y_pred_df], axis=1)

df_expanded = df_expanded[~df_expanded['Α.Μ.Δ'].isin(['212785', '192470', '193250'])]

# Correct erroneously recorded phosphorus (P) values in the extended dataset
df_expanded['P'] = df_expanded['P'].replace(1308.5, 130.9)
df_expanded['P'] = df_expanded['P'].replace(676.4, 67.5)
df_expanded['P'] = df_expanded['P'].replace(605.57, 60.5)

# Calculate part A based on given formula
partA = 0.278 * (df_expanded['S'] / 100) + 0.034 * (df_expanded['C'] / 100) + 0.022 * df_expanded['Organic matter'] - 0.018 * (
        (df_expanded['S'] / 100) * df_expanded['Organic matter']) - 0.027 * ((df_expanded['C'] / 100) * df_expanded['Organic matter']) - 0.584 * (
                (df_expanded['S'] / 100) * (df_expanded['C'] / 100)) + 0.078

# Calculate part B, enhancing part A with additional operations
partB = partA + 0.636 * partA - 0.107

# Calculate part C with a difZnrent set of operations and factors
partC = -0.251 * (df_expanded['S'] / 100) + 0.195 * (df_expanded['C'] / 100) + 0.011 * df_expanded['Organic matter'] + 0.006 * (
        (df_expanded['S'] / 100) * df_expanded['Organic matter']) - 0.027 * ((df_expanded['C'] / 100) * df_expanded['Organic matter']) + 0.452 * (
                (df_expanded['S'] / 100) * (df_expanded['C'] / 100)) + 0.299

# Enhance part C with additional operations to calculate part D
partD = partC + 1.283 * partC * partC - 0.374 * partC - 0.015

# Combine parts D and B and adjust with additional operations to calculate part E
partE = partD + partB - 0.097 * (df_expanded['S'] / 100) + 0.043

# Final calculation for FEB, scaling the result of (1 - partE) by 2
df_expanded['FEB'] = (1 - partE) * 2.65

# create columns with Phosphorus applied in kg/stremma for equilibrium concentrations 1, 2, 4, 6, and 10 mg/L
df_expanded['1ppm_applied'] = 30*(1/1000)*(1000/3)*2.29*(1000*0.15*df_expanded['FEB']*1000/1000000)
df_expanded['2ppm_applied'] = 30*(2/1000)*(1000/3)*2.29*(1000*0.15*df_expanded['FEB']*1000/1000000)
df_expanded['4ppm_applied'] = 30*(4/1000)*(1000/3)*2.29*(1000*0.15*df_expanded['FEB']*1000/1000000)
df_expanded['6ppm_applied'] = 30*(6/1000)*(1000/3)*2.29*(1000*0.15*df_expanded['FEB']*1000/1000000)
df_expanded['10ppm_applied'] = 30*(10/1000)*(1000/3)*2.29*(1000*0.15*df_expanded['FEB']*1000/1000000)

# convert applied P to mg/kg of soil
df_expanded['1ppm_applied'] = (df_expanded['1ppm_applied']*0.4364*1000000)/(df_expanded['FEB']*15*10000)
df_expanded['2ppm_applied'] = (df_expanded['2ppm_applied']*0.4364*1000000)/(df_expanded['FEB']*15*10000)
df_expanded['4ppm_applied'] = (df_expanded['4ppm_applied']*0.4364*1000000)/(df_expanded['FEB']*15*10000)
df_expanded['6ppm_applied'] = (df_expanded['6ppm_applied']*0.4364*1000000)/(df_expanded['FEB']*15*10000)
df_expanded['10ppm_applied'] = (df_expanded['10ppm_applied']*0.4364*1000000)/(df_expanded['FEB']*15*10000)

# convert from stremma to ha
df_expanded['1ppm_applied'] = 10 * df_expanded['1ppm_applied']
df_expanded['2ppm_applied'] = 10 * df_expanded['2ppm_applied']
df_expanded['4ppm_applied'] = 10 * df_expanded['4ppm_applied']
df_expanded['6ppm_applied'] = 10 * df_expanded['6ppm_applied']
df_expanded['10ppm_applied'] = 10 * df_expanded['10ppm_applied']

# Calculate percentage phosphorus adsorption for each corresponding application rate
df_expanded['PFP1'] = round(df_expanded['XGBoost_1ppm_adsorbed']*100/df_expanded['1ppm_applied'],1)
df_expanded['PFP2'] = round(df_expanded['XGBoost_2ppm_adsorbed']*100/df_expanded['2ppm_applied'],1)
df_expanded['PFP4'] = round(df_expanded['XGBoost_4ppm_adsorbed']*100/df_expanded['4ppm_applied'],1)
df_expanded['PFP6'] = round(df_expanded['XGBoost_6ppm_adsorbed']*100/df_expanded['6ppm_applied'],1)
df_expanded['PFP10'] = round(df_expanded['XGBoost_10ppm_adsorbed']*100/df_expanded['10ppm_applied'],1)

df_expanded.to_excel(f'final_output_percentage_adsorption.xlsx', index=False)
