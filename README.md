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
```bash
python lstm.py
```


## References
[Baseline Model](https://medium.com/@ozdogar/time-series-forecasting-using-lstm-pytorch-implementation-86169d74942e)  
[Value Prediction Network](https://notesonai.com/value+prediction+network)
