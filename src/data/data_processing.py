import pandas as pd

# Default threshold for rare categories (< DEFAULT_THRESHOLD is considered rare)
DEFAULT_THRESHOLD = 10

# Two kind of feature are used for training model: LAST_FAILURE_FEATURES (features of last failure) and MACHINE_FEATURES (features of machine)
LAST_FAILURE_FEATURES = ['Vùng thao tác','Mã xử lý', 'Mã Hiện tượng', 'Mã Nguyên nhân', 'Thời gian dừng máy (giờ)', 'Số người thực hiện', 'Ngày hoàn thành']
MACHINE_FEATURES = ['Tên thiết bị', 'Line', 'Ngày sản xuất', 'Điện áp tiêu thụ (V)']

# Dictionary of all columns and their categories (0 means continuous)   
ALL_COLUMNS_CATEGORIES = {
    'Tên thiết bị': ['Máy rửa', 'Máy kiểm tra bề mặt rỗ khí', 'OP1', 'OP2', 'OP3', 'OP4', 'OP5', 'OP6', 'OP7', 'OP8'],
    'Line': ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'],   
    'Vùng thao tác': [i for i in range(1, 28)],
    'Mã xử lý': [i for i in range(0, 8)],
    'Mã Hiện tượng': [i for i in range(0, 100)],
    'Mã Nguyên nhân': [i for i in range(0, 100)],
    'Thời gian dừng máy (giờ)': 0,
    'Số người thực hiện': 0,
    'Tuổi thọ thiết bị': 0,
    'Điện áp tiêu thụ (V)': 0,
}


def read_and_prepare_maintenance_data(filepath):
    """Read and preprocess the maintenance data."""
    df = pd.read_csv(filepath, usecols=['Số quản lý thiết bị', 'Vùng thao tác', 'Mã xử lý', 'Mã Hiện tượng', 
                                        'Mã Nguyên nhân', 'Thời gian dừng máy (giờ)', 'Số người thực hiện', 'Ngày hoàn thành'])
    df['Ngày hoàn thành'] = pd.to_datetime(df['Ngày hoàn thành'], format='%d-%m-%Y')
    df = df.sort_values(by=['Số quản lý thiết bị', 'Ngày hoàn thành'])
    df['Time Since Last Fix'] = df.groupby('Số quản lý thiết bị')['Ngày hoàn thành'].diff()
    df['Event'] = 1
    
    current_date = pd.to_datetime('today').normalize()
    for idx in df.groupby('Số quản lý thiết bị').tail(1).index:
        latest = df.loc[idx]
        censored_row = {
            'Số quản lý thiết bị': latest['Số quản lý thiết bị'],
            'Vùng thao tác': latest['Vùng thao tác'],
            'Mã xử lý': latest['Mã xử lý'],
            'Mã Hiện tượng': latest['Mã Hiện tượng'],
            'Mã Nguyên nhân': latest['Mã Nguyên nhân'],
            'Thời gian dừng máy (giờ)': latest['Thời gian dừng máy (giờ)'],
            'Số người thực hiện': latest['Số người thực hiện'],
            'Ngày hoàn thành': current_date,
            'Time Since Last Fix': current_date - latest['Ngày hoàn thành'],
            'Event': 0,
        }
        df = pd.concat([df, pd.DataFrame([censored_row])], ignore_index=True)

    df[LAST_FAILURE_FEATURES] = df.groupby('Số quản lý thiết bị')[LAST_FAILURE_FEATURES].shift(1)
    df = df.groupby('Số quản lý thiết bị').apply(lambda x: x.iloc[1:])
    return df.reset_index(drop=True)

def read_and_prepare_machine_data(filepath, maintenance_df):
    """Read and merge machine feature data with maintenance data."""
    current_date = pd.to_datetime('today').normalize()
    machine_df = pd.read_csv(filepath)
    merged_df = maintenance_df.merge(machine_df[MACHINE_FEATURES + ['Số quản lý thiết bị']], on='Số quản lý thiết bị', how='left')
    merged_df['Tuổi thọ thiết bị'] = (current_date - pd.to_datetime(merged_df['Ngày sản xuất'], format='%d-%m-%Y')).dt.days
    merged_df.drop(columns=['Ngày sản xuất', 'Số quản lý thiết bị', 'Ngày hoàn thành'], inplace=True)
    return merged_df

def process_categorical_columns(df, all_columns, threshold):
    """Group rare categories and one-hot encode categorical columns."""
    category_columns = [col for col, categories in all_columns.items() if categories != 0]
    used_categories = {}
    
    for col, categories in all_columns.items():
        if categories == 0:
            used_categories[col] = 0
            continue

        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < threshold].index
        df[col] = df[col].apply(lambda x: x if x in categories and x not in rare_categories else 'Other')
        
        unique_categories = df[col].unique().tolist()
        if 'Other' in unique_categories:
            unique_categories.remove('Other')
        used_categories[col] = unique_categories
        df[col] = pd.Categorical(df[col], categories=['Other'] + unique_categories)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=category_columns, drop_first=True)
    return df, used_categories

# Main
def prepare_training_data(maintenance_path, machine_path):
    """Main function to prepare training data."""
    # Read and preprocess data
    maintenance_df = read_and_prepare_maintenance_data(maintenance_path)
    full_df = read_and_prepare_machine_data(machine_path, maintenance_df)
    full_df['Time Since Last Fix'] = full_df['Time Since Last Fix'].dt.days
    full_df['Số người thực hiện'] = pd.to_numeric(full_df['Số người thực hiện'])

    # Process categorical columns
    processed_df, used_categories = process_categorical_columns(full_df, ALL_COLUMNS_CATEGORIES, DEFAULT_THRESHOLD)
    return processed_df, used_categories