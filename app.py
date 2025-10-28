lood-pattern analysis (Google Colab ready)
# Make sure to upload your dataset first, e.g., 'FLOOD DATA.csv'

import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, warnings
from scipy import stats
warnings.filterwarnings("ignore")

CSV_PATH = "/content/cleaned_flood_data.csv"  # <-- update if needed

# --- Load ---
df = pd.read_csv(CSV_PATH, encoding='latin1', low_memory=False)
print("Loaded:", CSV_PATH, "shape:", df.shape)
display(df.head())

# --- helper to detect likely columns by keywords ---
cols = [c.lower() for c in df.columns]
def find_col(keywords, cols=cols):
    for k in keywords:
        for i,c in enumerate(cols):
            if k in c:
                return df.columns[i]
    return None

date_col = find_col(['date','datetime','time','day'])
water_col = find_col(['water','level','wl','depth','height'])
area_col = find_col(['barangay','brgy','area','location','sitio'])
damage_inf_col = find_col(['infrastruct','infra','building'])
damage_agri_col = find_col(['agri','agriculture','crop','farm'])
damage_any_col = find_col(['damage','loss','estimated_damage','total_damage'])

print("Detected -> date:", date_col, "| water:", water_col, "| area:", area_col)
print("Damage cols:", damage_inf_col, damage_agri_col, damage_any_col)

# --- parse date and set as index ---
# Combine Date, Day, and Year columns for better parsing
if date_col and 'Day' in df.columns and 'Year' in df.columns:
    df['__combined_date'] = df['Date'].astype(str) + ' ' + df['Day'].astype(str) + ', ' + df['Year'].astype(str)
    date_col = '__combined_date'
elif date_col is None:
    df['__date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
    date_col = '__date'

df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
df = df.sort_values(by=date_col).reset_index(drop=True)
df = df.set_index(pd.DatetimeIndex(df[date_col]))
df = df.dropna(subset=[date_col]) # Drop rows where date parsing failed


# --- water level handling ---
if water_col is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No water-level or numeric column found. Add a water-level column.")
    water_col = numeric_cols[0]
    print("No explicit water column; using:", water_col)

df[water_col] = pd.to_numeric(df[water_col], errors='coerce')
df[water_col] = df[water_col].interpolate(method='linear', limit_direction='both')

# --- z-score & flood heuristic ---
df['zscore_water'] = stats.zscore(df[water_col].fillna(df[water_col].mean()))
df['is_outlier_water'] = df['zscore_water'].abs() > 3
occurrence_col = find_col(['flood','event','is_flood','flooded','occurrence'])
if occurrence_col:
    df['is_flood'] = df[occurrence_col].astype(bool)
else:
    threshold = df[water_col].mean() + 1.0 * df[water_col].std()
    df['is_flood'] = (df[water_col] >= threshold) | (df['zscore_water'].abs() > 1.5)

df['year'] = df.index.year

# --- damage columns (if present) ---
damage_cols = [c for c in [damage_inf_col, damage_agri_col, damage_any_col] if c is not None and c in df.columns]
damage_cols = list(dict.fromkeys(damage_cols))
if damage_cols:
    for c in damage_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace('[^0-9.-]','', regex=True), errors='coerce')
    total_damage_per_year = df.groupby('year')[damage_cols].sum().fillna(0)
else:
    total_damage_per_year = pd.DataFrame()

# --- aggregations ---
floods_per_year = df.groupby('year')['is_flood'].sum().astype(int)
avg_water_per_year = df.groupby('year')[water_col].mean()

if area_col and area_col in df.columns:
    most_affected = df[df['is_flood']].groupby(area_col)['is_flood'].sum().sort_values(ascending=False).head(10)
else:
    most_affected = None

# --- outputs dir ---
OUTDIR = "/content/flood_analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# Plot: water-level time series
plt.figure(figsize=(12,4))
plt.plot(df.index, df[water_col])
plt.title("Water level time series")
plt.xlabel("Date"); plt.ylabel(str(water_col))
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"water_level_timeseries.png")); plt.show()

# Plot: water-level with flood markers
plt.figure(figsize=(12,4))
plt.plot(df.index, df[water_col])
plt.scatter(df.index[df['is_flood']], df[water_col][df['is_flood']], s=20)
plt.title("Water level with flood event markers")
plt.xlabel("Date"); plt.ylabel(str(water_col))
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"water_level_with_flood_markers.png")); plt.show()

# Plot: floods per year
plt.figure(figsize=(8,4))
floods_per_year.plot(kind='bar')
plt.title("Flood occurrences per year")
plt.xlabel("Year"); plt.ylabel("Number of flood records")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"floods_per_year.png")); plt.show()

# Plot: avg water per year
plt.figure(figsize=(8,4))
avg_water_per_year.plot(kind='bar')
plt.title("Average water level per year")
plt.xlabel("Year"); plt.ylabel(f"Average {water_col}")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"avg_water_per_year.png")); plt.show()

# Plot: most affected areas (if found)
if most_affected is not None:
    plt.figure(figsize=(8,4))
    most_affected.plot(kind='bar')
    plt.title("Top affected areas by flood count")
    plt.xlabel("Area"); plt.ylabel("Flood count")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"most_affected_areas.png")); plt.show()

# Plot: total damage per year (if present)
if not total_damage_per_year.empty:
    plt.figure(figsize=(8,4))
    for c in total_damage_per_year.columns:
        plt.plot(total_damage_per_year.index, total_damage_per_year[c], marker='o')
    plt.title("Total damage per year (by damage column)")
    plt.xlabel("Year"); plt.ylabel("Damage (dataset units)")
    plt.legend(total_damage_per_year.columns)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"total_damage_per_year.png")); plt.show()

# --- SARIMA attempt ---
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

series = df[water_col].resample('M').mean().fillna(0) # Fill missing months with 0
if len(series) >= 12:
    split = int(len(series)*0.😎
    train = series.iloc[:split]; test = series.iloc[split:]
    best_aic = 1e18; best_res=None; best_order=None
    for p in range(0,2):
        for d in range(0,2):
            for q in range(0,2):
                for P in range(0,2):
                    for D in range(0,2):
                        for Q in range(0,2):
                            try:
                                mod = SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,12),
                                              enforce_stationarity=False, enforce_invertibility=False)
                                res = mod.fit(disp=False)
                                if res.aic < best_aic:
                                    best_aic = res.aic; best_res = res; best_order=((p,d,q),(P,D,Q,12))
                            except: pass
    if best_res is not None:
        pred = best_res.get_prediction(start=test.index[0], end=test.index[-1], dynamic=False)
        forecast = pred.predicted_mean
        mae = mean_absolute_error(test, forecast); mse = mean_squared_error(test, forecast)
        print("SARIMA best order:", best_order, "AIC:", best_aic)
        print("MAE:", mae, "MSE:", mse)
        plt.figure(figsize=(10,4))
        plt.plot(train.index, train, label='Train'); plt.plot(test.index, test, label='Test')
        plt.plot(forecast.index, forecast, label='SARIMA Forecast'); plt.legend()
        plt.title("SARIMA: Actual vs Forecast"); plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR,"sarima_actual_vs_forecast.png")); plt.show()
else:
    print("Not enough monthly data for SARIMA (need >=12 aggregated months).")

summary_df = pd.DataFrame({
    "year": floods_per_year.index,
    "floods_per_year": floods_per_year.values,
    "avg_water_per_year": avg_water_per_year.values
})
summary_df.to_csv(os.path.join(OUTDIR,"summary_per_year.csv"), index=False)
print("Outputs saved to:", OUTDIR)
print("Files:", os.listdir(OUTDIR))
