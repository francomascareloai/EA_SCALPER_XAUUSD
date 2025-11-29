# Gold Price Prediction in Forex - XAUUSD Machine Learning

Repository: Gold-Price-Trend-Prediction-in-Forex
Focus: XAU/USD prediction using LSTM, Naive Bayes, Random Forest

# Gold Price Trend Prediction in Forex
Gold price prediction, XAU/USD, Forex market, LSTM, NaÃ¯ve Bayes, Random Forest, feature engineering, time series forecasting, deep learning, machine learning.


---

# Forex Market Prediction with Deep Learning

Repository: Forex-Market-Prediction-DeepL
Focus: FEDformer, Transformers, LSTM for Forex prediction

# Forex Market Prediction using Deep Learning

## Overview
This project is an experiment to apply deep learning techniques to predict future market prices in the Forex market. The goal is to leverage advanced time series forecasting models to enhance predictive accuracy and inform trading strategies.

## Installation and Setup
To run this project, follow the steps below:

### 1. Install Python and Create a Virtual Environment
Ensure you have Python 3.8 or later installed. You can check your version by running:

```bash
python --version
```

#### Create a Virtual Environment
1. Navigate to the project folder:

   ```bash
   cd path/to/project
   ```

2. Create a virtual environment:

   ```bash
   python -m venv forex_env
   ```

3. Activate the virtual environment:

   - **Windows**:
     ```bash
     forex_env\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source forex_env/bin/activate
     ```

### 2. Install Dependencies
Once inside the virtual environment, install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Download Data
The dataset for this project includes historical forex market prices and can be obtained from relevant data sources.

For benchmark datasets, you can refer to:
- [Autoformer Repository](https://github.com/thuml/Autoformer)
- [Informer Repository](https://github.com/zhouhaoyi/Informer2020)

Ensure your dataset is placed inside the `data/` folder before proceeding.

### Changing the Dataset
To change the dataset and include a new dataset, you can run the `data_downloading_and_feature_extraction0.ipynb` notebook and modify the parameters within the notebook accordingly.

## Running the Experiment
To execute the experiment, follow these steps:

1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to the project folder and open `fedformer6.ipynb`.

3. Run the notebook cells sequentially to execute the Forex market prediction script.

## Model Architecture
This model utilizes an advanced time series forecasting transformer-based architecture. Below is the structure of the model:

|![Model Structure](https://user-images.githubusercontent.com/44238026/171341166-5df0e915-d876-481b-9fbe-afdb2dc47507.png)|
|:--:| 
| *Figure 1. Model architecture used for time-series forecasting* |

The model is designed to efficiently process long-term dependencies in time series data by leveraging frequency-enhanced decomposition techniques.

## Experiment Execution
Once the setup is complete:

- Open the Jupyter notebook (`fedformer6.ipynb`).
- Execute all cells sequentially.
- The model will train on the dataset and provide market price predictions.

## Citation
If this experiment is useful to you, consider referring to related work:

```
@inproceedings{zhou2022fedformer,
  title={Frequency enhanced decomposed transformer for long-term series forecasting},
  author={Zhou, Tian and Ma, Ziqing and Wen, Qingsong and Wang, Xue and Sun, Liang and Jin, Rong},
  booktitle={Proc. 39th International Conference on Machine Learning (ICML 2022)},
  location = {Baltimore, Maryland},
  year={2022}
}
```

## Further Reading
For further understanding of transformers in time-series forecasting:
- Qingsong Wen, Tian Zhou, et al. "Transformers in time series: A survey." [arXiv preprint](https://arxiv.org/abs/2202.07125).

## Contact
For questions regarding this experiment, please reach out to the project contributors or refer to the linked repositories:

- [Autoformer Repository](https://github.com/thuml/Autoformer)
- [Informer Repository](https://github.com/zhouhaoyi/Informer2020)
- [Multivariate Time Series Data](https://github.com/laiguokun/multivariate-time-series-data)

---
This document provides a comprehensive guide to setting up and running the experiment. The model is built to offer insights into market trends using advanced deep learning techniques.


