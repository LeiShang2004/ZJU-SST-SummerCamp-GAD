import os
import pandas as pd
from datetime import datetime

def merge_csv_files():
    results_dir = './'

    all_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.csv')]

    for f in all_files:
        print(f"- {f}")

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
            continue

    consolidated_df = pd.concat(df_list, ignore_index=True)

    consolidated_df.sort_values(by=['Dataset', 'Timestamp'], inplace=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'consolidated_results_{timestamp}.csv'

    try:
        consolidated_df.to_csv(output_filename, index=False)
        print(f"\n保存到: {output_filename}")
    except Exception as e:
        print(f"\n出错: {e}")


if __name__ == '__main__':
    merge_csv_files()