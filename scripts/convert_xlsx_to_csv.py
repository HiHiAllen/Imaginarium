import pandas as pd
import os
import numpy as np

# 输入文件路径
input_file = 'asset_data/imaginarium_asset_info_with_render_images.xlsx'
# 输出文件名称
output_csv = 'asset_data/imaginarium_asset_info.csv'
output_xlsx = 'asset_data/imaginarium_asset_info.xlsx'
def main():
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Reading file: {input_file}")
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
    except Exception as e:
        print(f"Failed to read excel: {e}")
        return

    print(f"Total rows: {len(df)}")    
    print(f"\nSaving to {output_csv}...")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 兼容 Excel 打开 CSV 中文乱码
    
    print(f"Saving to {output_xlsx}...")
    df.to_excel(output_xlsx, index=False, engine='openpyxl')
    
    print("Done.")

if __name__ == "__main__":
    main()

