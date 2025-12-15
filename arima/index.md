---

layout: default

title: Air Passengers Time-Series Prediction (ARIMA)

permalink: /arima/

---

#### This project is in development

## Goals and objectives:

The business objective is to predict monthly air passenger volumes using historical observations.  The Autoregressive Integrated Moving Average (ARIMA) technique was selected for this purpose as it is a powerful time-series analysis tool that can factor in seasonality and trends within time-series data, to determine patterns and build a prediction model.

ARIMA is effectively a model comprising of three components: AR (AutoRegressive), I (Integrated) and MA (Moving Average), which collectively operate on the time-series data.

The model built reached an accuracy of...

## Application:  

The Autoregressive Integrated Moving Average (ARIMA) model is a powerful statistical technique for time series forecasting, making it invaluable across various business sectors. By decomposing and modelling the trend, seasonality, and random noise in historical data, ARIMA provides precise, short-to-medium-term predictions that enable proactive strategic and operational decisions.

ARIMA techniques can be applied to many real-world scenarios, yielding benefits across numerous industries by providing insights and supporting decision-making.  Example benefits include the following:

* **Finance: Risk Management and Market Prediction** - ARIMA is used to forecast highly volatile and continuous data streams, supporting investment and risk management strategies.
  * Short-Term Price Forecasting - predicting stock prices and exchange rates to inform trading decisions and setting market positions
  * Volatility Forecasting - modelling the variation of financial time series, to predict prediod of high market risk
  * Financial Metric Forecasting - Predicting key economic indicators like inflation rates or interest rate movements to guide long-term financial planning and capital allocation.
* **Manufacturing: Production & Quality Optimisation** - ARIMA models drive efficiency and quality control in manufacturing by providing accurate, timely projections of resource needs and process stability.
  * Raw Material Procurement	- Forecasting the future demand for key raw materials with high accuracy, allowing purchasing managers to optimise order quantities, negotiate better prices, and minimize storage costs.
  * Inventory & Production Scheduling - Used to predict the required inventory levels for goods, leading to optimised production runs, reduced idle time, and lower costs associated with overstocking.
  * Process Control	- Monitoring critical process variables (e.g., temperature, pressure, chemical concentration) to predict when the process is likely to fail or experience issues based on historical patterns, supporting preventative maintenance.
* **Retail: Demand Planning and Supply Chain Efficiency** - ARIMA is used to capture seasonality is key to maximizing sales and managing costs.
  * Demand Forecasting	- Accurately predicting weekly or monthly sales volume for individual products or entire categories, crucial for seasonal events like holidays (e.g., Christmas) where demand spikes significantly.
  * Preventing Stockouts/Oversupply -	Precise forecasts ensures popular items are in stock to meet demand (improving customer satisfaction) and prevents over-ordering, reducing the capital tied up in slow-moving inventory.
  * Staff Scheduling -	Forecasting customer traffic or required checkout volume by hour or day, enabling managers to align staff levels with anticipated demand, reducing labour costs and wait times.
* **Technology: Capacity Planning and Service Reliability** - ARIMA is vital for managing infrastructure and service quality in response to fluctuating usage.
  * Server Load & Traffic Prediction	- Forecasting website traffic, API call volume, or server CPU load to predict when capacity will be exceeded. This is essential for proactive scaling of cloud resources (AWS, Azure, etc.) to prevent service outages.
  * Resource Allocation	- Predicting the demand for storage, bandwidth, or computing resources on a weekly or monthly basis to guide hardware procurement and capacity planning, optimising capital expenditure spending.
  * System Performance	- Forecasting trends in system metrics like network latency or error rates. If the forecast shows a steady upward trend in latency, it flags a need for system optimisation before performance degrades noticeably.

## Methodology:  

A workflow was develped in Python using statsmodels, scikit-learn, pandas and numpy libraries, with Matplotlib and Seasborn packages for visualisation.

The data of observed air passengers was split into a training set and a testing set, where the first 80% of observations formed the training set, and the latest 20% of observations formed the test set.

#### Descriptive Analysis
The original data was analysed to understand and visualise any poentential trends, patterns and seasonality.

#### Integrated / Differencing (Stationarity)
ARIMA models are designed to handle non-stationary time series by incorporating differencing into the model itself. The “I” in ARIMA stands for Integrated, which refers to the differencing step that makes the series stationary.  Stationarity is required for AR (AutoRegressive) and MA (Moving Average) components.  While ARIMA handles stationarity internally via differencing, it may be required to apply pre-transformation to the data prior to applying the ARIMA methods, for example the data has variance instability (e.g., heteroscedasticity).


## Results and conclusions:

The overall training set has a mean of 239.95, with a standard deviation of 91.35 (variance = 8,344.42), however a plot of the points (both training and testing data) demonstrate a seasonal pattern seemingly with an increasing trend.  Additional plots were generated to show the moving average and moving variance of the training data.  As shown below these further highlight the overall trend of increasing passenger volumes, and also indicate an increase in variance over time.

The following shows the plot of both the training and testing datsets:

![dataplot](arima_data_split.png)

The plots below show the moving average and moving variance of the training data, both suggesting non-stationarity of the data which will next be more formally tested.

![train_mean](arima_train_mean.png)
![train_variance](arima_train_variance.png)

The Augmented Dickey-Fuller (ADF) test was applied to the original training data to test for stationarity, where the null hypothesis (H₀) is that the data is non-stationary.  This produced an ADF test statistic = -0.3569, which produces a p-value of 0.9171, therefore there is insufficient evidence to reject to null hypothesis and there is evidence that the data is non-stationary, and differencing is required.

Results from the project related to the business objective.

### Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yeild tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/ARIMA_AirPassengers_v3.py)
