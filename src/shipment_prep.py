import pandas as pd
import numpy as np


PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):

    #Drop duplicates
    df = raw.copy().drop_duplicates().reset_index(drop=True)

    #Clean service_type
    df['service_type']= df['service_type'].str.strip().str.title()

    #Clean payment_mode
    df['payment_mode'] = df['payment_mode'].fillna("Unknown")

    #Clean declared_value
    df['declared_value'] = df['declared_value'].str.replace('₹', '', regex=False).str.replace(',', '', regex=False)
    df['declared_value_inr'] = pd.to_numeric(df['declared_value'], errors='coerce')
    df.drop(columns=['declared_value'], inplace=True)

    #Clean dispatch_time
    df['dispatch_time'] = pd.to_datetime(df['dispatch_time'], errors='coerce')

    #Clean parcel_weight_kg
    df['parcel_weight_kg'] = df['parcel_weight_kg'].where(df['parcel_weight_kg'] <= 100, np.nan)

    if medians is None:
        median1 = float(df['distance_km'].median())
        median2 = float(df['parcel_weight_kg'].median())
        median3 = float(df['declared_value_inr'].median())
        df['distance_km'] = df['distance_km'].fillna(median1)
        df['parcel_weight_kg'] = df['parcel_weight_kg'].fillna(median2)
        df['declared_value_inr'] = df['declared_value_inr'].fillna(median3)
        medians = {'distance_km': median1, 'parcel_weight_kg': median2, 'declared_value_inr': median3}
    else:
        df['distance_km'] = df['distance_km'].fillna(medians['distance_km'])
        df['parcel_weight_kg'] = df['parcel_weight_kg'].fillna(medians['parcel_weight_kg'])
        df['declared_value_inr'] = df['declared_value_inr'].fillna(medians['declared_value_inr'])
        medians = medians
    
    return (df, medians)

def build_features(clean, hubs):
    df= pd.merge(clean, hubs, how='left', on='hub_id')
    df['promised_hours'] = df['service_type'].map(PROMISED_HOURS)
    df['is_cod'] = (df['payment_mode'] == 'COD').astype(int)
    df['is_weekend'] = (df['dispatch_time'].dt.dayofweek >= 5).astype(int)
    df['dispatch_hour'] = df['dispatch_time'].dt.hour.astype(int)
    df['log_declared_value'] = np.log(df['declared_value_inr']).astype(float)
    return df