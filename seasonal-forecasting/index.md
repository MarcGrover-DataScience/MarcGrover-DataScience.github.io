---

layout: default

title: Electricity Demand Forecasting (Prophet)

permalink: /seasonal-forecasting/

---

# This project is in development

## Goals and objectives:

The business objective is to determine how accurately an energy retailer's procurement team can forecast daily electricity demand 90 and 365 days ahead, the horizons over which forward power purchases are planned, and how large a buffer margin it should hold against the forecast error that remains. The decision at stake is concrete: a retailer that buys forward exactly the forecast volume is exposed on every day that demand exceeds it, while one that over-buys is left holding surplus volume, so the size of the forecast error directly sets the size of the buffer. The project also addresses two supporting questions a decision-maker would raise before trusting a forecasting model: whether Prophet's explicit treatment of trend changes, weekly and annual seasonality and public holidays delivers measurably better forecasts than classical alternatives, and whether additional external data, in this case wind and solar generation, is worth the cost of acquiring and maintaining. The dataset is the Open Power System Data (OPSD) daily German electricity series: 4,383 days of national consumption (GWh) from 1 January 2006 to 31 December 2017, with wind generation available from 2010 and solar generation from 2012. The retailer's own demand is assumed to be proportional to national consumption.

The analytical scope is deliberately broader than a single model fit. The series is split chronologically into ten training years (2006 to 2015, 3,652 days) and two test years (2016 to 2017, 731 days), and every modelling decision is made on the training years alone, consistent with the chronological, walk-forward validation set out on the Data Science Workflow page. Exploratory analysis characterises the weekly cycle (Sunday consumption 17.3% below the calendar-year mean), the annual cycle (January 9.4% above the mean, August 6.6% below), the fall of 27 to 28% on Christmas Day, Boxing Day and New Year's Day, and two features of the data that shape the modelling: a sustained level shift of +8.7% at the turn of 2013/14, for which a change in the source's coverage is a candidate explanation, and an unexplained shortfall between April and September 2009. Three classical baselines of increasing sophistication (a seasonal naive forecast, a weekly SARIMA, and a SARIMAX with annual Fourier terms, holiday indicators and a level-step indicator) are then compared with Prophet, whose hyperparameters are tuned by nine walk-forward cross-validation folds inside the training data. A controlled experiment adds wind and solar generation to Prophet as external regressors, benchmarked against placebo regressors and paired block-bootstrap intervals so that a small change in error is not mistaken for signal. Finally, a bridge analysis repeats the ARIMA project's Air Passengers forecast with Prophet on an identical split, giving a like-for-like comparison of the two techniques.

This project is positioned differently from the two existing time-series projects in this portfolio. The Moving Averages project smooths the S&P 500 to characterise what has already happened: its best model, the 30-day weighted moving average, reports an MAE of $51.46 (MAPE 1.83%), but that figure measures how closely a lagging average tracks the price series it was computed from, not how well it predicts dates it has never seen. Here every forecast is made for days beyond the last observation, and accuracy is judged on 731 genuinely unseen days. The ARIMA project forecast 144 monthly Air Passengers observations with a single seasonal cycle, where ARIMA(12,2,12) on Box-Cox-transformed data reached an MAE of 14.89 thousand passengers, an R² of 0.947 and a MAPE of 3.46% on a 29-month holdout, with the model orders informed by autocorrelation analysis and the transformation chosen to handle seasonal swings that grow with the series. This project moves to daily data with two interacting seasonal cycles, public holidays and a level shift, structure that a seasonal ARIMA cannot express on its own and that Prophet represents explicitly. The ARIMA project named Prophet as the natural comparator in its Next Steps, and this project takes that up by running both techniques on the identical 115 / 29 month split.

The analysis shows that Prophet, once its configuration is tuned on the training years, produces materially better forecasts than the classical alternatives on this series. On the 731 held-out days it achieves an MAE of 31.5 GWh/day (MAPE 2.30%, R² 0.92), against 41.9 GWh/day (3.02%) for the SARIMAX baseline that was given the same holiday and level-step information, 40.3 GWh/day (3.05%) for an out-of-the-box Prophet, 47.0 GWh/day (3.50%) for a seasonal naive forecast and 87.0 GWh/day (6.20%) for a weekly-only SARIMA. A paired block bootstrap puts the advantage over SARIMAX at 10.4 GWh/day (95% interval 6.4 to 14.0), whereas SARIMAX and the out-of-the-box Prophet cannot be distinguished from each other, so the gain comes from the tailored configuration (custom holidays, a level-step regressor and tuned trend flexibility) rather than from Prophet alone. Cross-validated error rises with horizon, from a MAPE of 3.26% over the first 90 days to 4.18% over a full year, a range that includes the 2013/14 level shift, which the earlier folds could not see coming. Wind and solar generation add nothing detectable even when their future values are supplied perfectly: the change in MAE is within about ±1 GWh/day and inside the noise floor set by placebo regressors. On the Air Passengers series, tuned Prophet (MAE 16.2) and ARIMA (14.9) cannot be told apart (difference +1.3, 95% interval −2.2 to +7.2), and in both techniques the decisive choice is how to handle seasonality that grows with the series: multiplicative rather than additive seasonality improves Prophet's MAE by 15.1 thousand passengers. As elsewhere in this portfolio, the purpose is not to demonstrate business-grade accuracy but to show a rigorous and honest forecasting workflow, in which the value lies in the methodology and the insight it produces.

## Application:  

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 


## Methodology:  

Details of the methodology applied in the project.

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

Next steps based on current results and conclusions from above and suggested follow-up actions, analysis etc.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/Prophet_Seasonal_Forecasting_OPSD_v6.py)
