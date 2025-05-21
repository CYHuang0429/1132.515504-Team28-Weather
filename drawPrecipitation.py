import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path: str) -> pd.Series:
    """
    讀取 CSV，並回傳「降雨量」欄位的數值序列（去除無效值）。
    """
    df = pd.read_csv(path)
    # 將 Precipitation 欄位轉成數值，無法轉換的設為 NaN，並去除
    df['Precipitation'] = pd.to_numeric(df['Precipitation'], errors='coerce')
    return df['Precipitation'].dropna()

def plot_histogram(precip: pd.Series, bin_width: float = 0.1):
    """
    繪製降雨量直方圖，bin_width 單位為毫米。
    """
    # 建立 bin 邊界
    bins = np.arange(0, precip.max() + bin_width, bin_width)
    
    plt.figure(figsize=(8, 4))
    plt.hist(precip, bins=bins, edgecolor='black')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Frequency')
    plt.title(f'降雨量分布 (每 {bin_width} mm 為一組)')
    plt.tight_layout()
    plt.savefig('precip_histogram.png', dpi=150)
    plt.show()

def plot_boxplot(precip: pd.Series):
    """
    繪製降雨量箱線圖，用於檢視離群值與分位數。
    """
    plt.figure(figsize=(8, 2))
    plt.boxplot(precip, vert=False, showfliers=True)
    plt.xlabel('Precipitation (mm)')
    plt.title('降雨量箱線圖')
    plt.tight_layout()
    plt.savefig('precip_boxplot.png', dpi=150)
    plt.show()

def main():
    # 1. 讀取並前處理資料
    data_path = 'Master_Hsinchu.csv'  # 如果你的檔案放在其他路徑，請修改此處
    precip = load_data(data_path)

    # 2. 列印基本統計量
    print("降雨量描述統計：")
    print(precip.describe(), end='\n\n')

    # 3. 繪製直方圖（3 mm 區間）
    plot_histogram(precip, bin_width=0.1)

    # 4. 繪製箱線圖
    plot_boxplot(precip)

if __name__ == '__main__':
    main()
