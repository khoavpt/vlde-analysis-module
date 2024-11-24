import pandas as pd

from ..data import data_processing

def fetch_raw_machine_data(machine_number, df1, df2):
    """
    Fetch raw data for a specific machine number from machine and maintenance data.
    """
    # Fetch machine data for the machine
    machine_feature =['Tên thiết bị', 'Line', 'Ngày sản xuất', 'Điện áp tiêu thụ (V)']
    machine_data = df1.loc[df1['Số quản lý thiết bị'] == machine_number][machine_feature].to_dict(orient='records')[0]
    machine_data['Tuổi thọ thiết bị'] = (pd.to_datetime('today').normalize() - pd.to_datetime(machine_data['Ngày sản xuất'], format='%d-%m-%Y')).days
    machine_data.pop('Ngày sản xuất', None)

    # Fetch the last fix data for the machine
    last_fix_feature = ['Vùng thao tác','Mã xử lý', 'Mã Hiện tượng', 'Mã Nguyên nhân', 'Nguyên nhân gốc (number)', 'Thời gian dừng máy (giờ)', 'Số người thực hiện']

    fix_data = df2.loc[df2['Số quản lý thiết bị'] == machine_number]
    if fix_data.empty:
        raise ValueError(f"No data found for machine {machine_number}")
    
    # Get the most recent record for the machine
    fix_data = fix_data.iloc[-1][last_fix_feature].to_dict()

    return {**machine_data, **fix_data}


def process_raw_machine_data(raw_data, known_categories):
    """
    Transform raw data into a format suitable for the model.
    """
    # Convert raw data to a DataFrame
    df = pd.DataFrame([raw_data])

    # Initialize a dictionary for processed data
    processed_data = {}

    # Process each column in known_categories
    for column, categories in known_categories.items():
        if categories == 0:
            # Include non-categorical columns directly
            processed_data[column] = df[column].values
        else:
            # One-hot encode categorical features based on known categories
            for category in categories:
                processed_data[f"{column}_{category}"] = (df[column] == category).astype(int).values
    # Convert processed data dictionary to a DataFrame
    processed_df = pd.DataFrame(processed_data)

    # Align with expected columns (all known categories)
    expected_columns = []
    for column, categories in known_categories.items():
        if categories == 0:
            expected_columns.append(column)
        else:
            expected_columns.extend([f"{column}_{category}" for category in categories])

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0  # Add missing column as 0

    # Reorder columns to match expected order
    processed_df = processed_df[expected_columns]

    return processed_df