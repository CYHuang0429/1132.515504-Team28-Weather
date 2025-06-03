# 1132.515504-Team28-Weather
NYCU 1132.515504 Final Project (Team 28)  
Topic: NYCU campus weather forecast  
Members:  
113550182 黃禎鈺  
113550123 蔡承軒  
112550109 楊榮竣 

## Specifications
This is a project about improving the spacial accuracy of weather forecasts, and learn the local Hsinchu weather data with AI.  
The file structure for this project is as follows:

```
1132.515504-Team28-Weather
├── Data
│   ├── EastHsinchu // unused data
│   └── Hsinchu // the dataset we used to train
├── Figure // some results it profuced
├── Masters
│   └── Master_Hsinchu.csv // our master data
├── lstm.py // our main model
├── baseline.py // baseline model
├── data_analysis.py // code for processing raw data into master files
└── README.md
```

## Dependencies
Run the following command to install all dependencies:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

Run for the model
(Running python 3.12.6)
```bash
python lstm.py
```

## Execution Result
This was the result we've achieved, note that although this slightly differs from the result shown in the slides and the video, both are valid execution results and is completely generated from our code.
``` bash
=== Classification Report on Test Set ===
              precision    recall  f1-score   support

     No Rain       0.99      0.99      0.99      7831
        Rain       0.73      0.73      0.73       402

    accuracy                           0.97      8233
   macro avg       0.86      0.86      0.86      8233
weighted avg       0.97      0.97      0.97      8233


=== Multi-Target Evaluation on All Test Samples ===
AirTemperature  → MAE: 0.61, RMSE: 0.85, R²: 0.9798
Precipitation   → MAE: 0.15, RMSE: 1.11, R²: 0.2481
WindSpeed       → MAE: 0.46, RMSE: 0.61, R²: 0.7505
=== Rainy-Hours Precipitation Regression ===
MAE (雨天时段): 2.48 mm
RMSE(雨天时段): 4.94 mm
```


## References
[Baseline Model](https://medium.com/@ozdogar/time-series-forecasting-using-lstm-pytorch-implementation-86169d74942e)  
[Value Prediction Network](https://notesonai.com/value+prediction+network)
