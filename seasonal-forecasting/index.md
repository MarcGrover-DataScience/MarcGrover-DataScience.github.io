---

layout: default

title: Electricity Demand Forecasting (Prophet)

permalink: /seasonal-forecasting/

---

# This project is in development

## Goals and objectives:

The business objective is to determine how accurately an energy retailer's procurement team can forecast daily electricity demand 90 and 365 days ahead, the horizons over which forward power purchases are planned, and how large a buffer margin it should hold against the forecast error that remains. The decision at stake is concrete: a retailer that buys forward exactly the forecast volume is exposed on every day that demand exceeds it, while one that over-buys is left holding surplus volume, so the size of the forecast error directly sets the size of the buffer. 

The project also addresses two supporting questions a decision-maker would raise before trusting a forecasting model: whether **Prophet's** explicit treatment of trend changes, weekly and annual seasonality and public holidays delivers measurably better forecasts than classical alternatives, and whether additional external data, in this case wind and solar generation, is worth the cost of acquiring and maintaining. The dataset is the Open Power System Data (OPSD) daily German electricity series: 4,383 days of national consumption (GWh) from 1 January 2006 to 31 December 2017, with wind generation available from 2010 and solar generation from 2012. The retailer's own demand is assumed to be proportional to national consumption.

The analytical scope is deliberately broader than a single model fit. The series is split chronologically into ten training years (2006 to 2015, 3,652 days) and two test years (2016 to 2017, 731 days), and every modelling decision is made on the training years alone, consistent with the chronological, walk-forward validation set out on the [Data Science Workflow](/data-science-workflow/) page. Exploratory analysis characterises the weekly cycle (Sunday consumption 17.3% below the calendar-year mean), the annual cycle (January 9.4% above the mean, August 6.6% below), the fall of 27 to 28% on Christmas Day, Boxing Day and New Year's Day, and two features of the data that shape the modelling: a sustained level shift of +8.7% at the turn of 2013/14, for which a change in the source's coverage is a candidate explanation, and an unexplained shortfall between April and September 2009. 

Three classical baselines of increasing sophistication (a **seasonal naive forecast**, a **weekly SARIMA**, and a **SARIMAX** with annual Fourier terms, holiday indicators and a level-step indicator) are then compared with Prophet, whose hyperparameters are tuned by nine walk-forward cross-validation folds inside the training data. A controlled experiment adds wind and solar generation to Prophet as external regressors, benchmarked against placebo regressors and paired block-bootstrap intervals so that a small change in error is not mistaken for signal. Finally, a bridge analysis repeats the ARIMA project's Air Passengers forecast with Prophet on an identical split, giving a like-for-like comparison of the two techniques.

This project is positioned differently from the two existing time-series projects in this portfolio. The [Moving Averages](/moving-averages/) project smooths the S&P 500 to characterise what has already happened: its best model, the 30-day weighted moving average, reports an MAE of $51.46 (MAPE 1.83%), but that figure measures how closely a lagging average tracks the price series it was computed from, not how well it predicts dates it has never seen. Here every forecast is made for days beyond the last observation, and accuracy is judged on 731 genuinely unseen days. The [ARIMA](/arima/) project forecast 144 monthly Air Passengers observations with a single seasonal cycle, where ARIMA(12,2,12) on Box-Cox-transformed data reached an MAE of 14.89 thousand passengers, an R² of 0.947 and a MAPE of 3.46% on a 29-month holdout, with the model orders informed by autocorrelation analysis and the transformation chosen to handle seasonal swings that grow with the series. This project moves to daily data with two interacting seasonal cycles, public holidays and a level shift, structure that a seasonal ARIMA cannot express on its own and that Prophet represents explicitly. The ARIMA project named Prophet as the natural comparator in its Next Steps, and this project takes that up by running both techniques on the identical 115 / 29 month split.

The analysis shows that Prophet, once its configuration is tuned on the training years, produces materially better forecasts than the classical alternatives on this series. On the 731 held-out days it achieves an MAE of 31.5 GWh/day (MAPE 2.30%, R² 0.92), against 41.9 GWh/day (3.02%) for the **SARIMAX** baseline that was given the same holiday and level-step information, 40.3 GWh/day (3.05%) for an **out-of-the-box Prophet**, 47.0 GWh/day (3.50%) for a **seasonal naive forecast** and 87.0 GWh/day (6.20%) for a weekly-only **SARIMA**. 

A paired block bootstrap puts the advantage over SARIMAX at 10.4 GWh/day (95% interval 6.4 to 14.0), whereas SARIMAX and the out-of-the-box Prophet cannot be distinguished from each other, so the gain comes from the tailored configuration (custom holidays, a level-step regressor and tuned trend flexibility) rather than from Prophet alone. 

Cross-validated error rises with horizon, from a MAPE of 3.26% over the first 90 days to 4.18% over a full year, a range that includes the 2013/14 level shift, which the earlier folds could not see coming. Wind and solar generation add nothing detectable even when their future values are supplied perfectly: the change in MAE is within about ±1 GWh/day and inside the noise floor set by placebo regressors. 

On the Air Passengers series, tuned Prophet (MAE 16.2) and ARIMA (14.9) cannot be told apart (difference +1.3, 95% interval −2.2 to +7.2), and in both techniques the decisive choice is how to handle seasonality that grows with the series: multiplicative rather than additive seasonality improves Prophet's MAE by 15.1 thousand passengers. As elsewhere in this portfolio, the purpose is not to demonstrate business-grade accuracy but to show a rigorous and honest forecasting workflow, in which the value lies in the methodology and the insight it produces.

## Application:  

Prophet is an open-source forecasting procedure developed by Meta's Core Data Science team (Taylor and Letham, 2018). It fits an additive regression model, or a multiplicative one where seasonal swings grow with the level of the series, built from three components: a piecewise-linear trend whose slope is allowed to change at automatically selected changepoints; seasonal patterns of any period, represented as Fourier series (here weekly and annual); and holiday effects specified through an analyst-supplied calendar. External regressors can be added, the model is fitted by optimisation in Stan, and prediction intervals are simulated from the uncertainty in the trend. This structure suits series dominated by calendar effects, with strong weekly and annual cycles, public holidays and occasional trend changes. Its practical benefits are interpretability and control: each component can be plotted and explained to a non-technical stakeholder, missing observations are tolerated, and a domain expert can supply knowledge directly (a holiday calendar, a known level shift) instead of relying on the optimiser to discover it. Its limitation, visible in this project's residual diagnostics, is that it does not model short-run autocorrelation, so it is best matched to medium and long horizons rather than next-day forecasting.

The business application of Prophet spans any domain where demand follows a calendar:

⚡ **Energy and utilities:**

**Forward procurement and demand forecasting**: as demonstrated in this project, retailers and traders forecast daily demand months ahead to set forward purchase volumes and hedging positions, where the weekly and annual cycles and holiday effects that Prophet isolates (a fall of 27 to 28% on Christmas Day) translate directly into volume decisions.  
**Grid and capacity planning**: system operators use long-horizon demand forecasts to plan generation capacity, maintenance windows and reserve margins, where an interpretable decomposition into trend and seasonality helps explain to regulators and planners why forecast peak demand is rising or falling.

🏥 **Healthcare:**

**Emergency department attendance and bed occupancy**: hospitals forecast daily attendances, which follow weekly rhythms, winter pressures and bank-holiday effects, to set staffing rosters and bed capacity weeks in advance.  
**Pharmacy and clinical supply demand**: seasonal illness and holiday opening patterns shape demand for medicines and consumables, and forecasts that reflect those calendars reduce both stock-outs and waste.

🚆 **Transport and logistics**:

**Passenger demand and capacity planning**: rail, bus and airline operators forecast ridership with commuting patterns by day of week, school-holiday peaks and public-holiday troughs, the same structure as the Air Passengers series revisited in this project, to schedule services and set fares.  
**Parcel and freight volumes**: courier and logistics networks forecast daily volumes with sharp annual peaks, such as the run-up to Christmas, to plan vehicles, depot capacity and temporary staffing.

🏨 **Hospitality and leisure**:

**Occupancy and booking demand**: hotels forecast daily occupancy, shaped by weekends, school holidays and local events, which Prophet accommodates through custom holiday windows, to inform pricing and staffing.  
**Visitor numbers for venues and attractions**: operators forecast footfall driven by weekends, holiday periods and the annual season to plan opening hours, staffing and capacity, where an explicit holiday calendar captures the days on which normal patterns break.

## Methodology:  

The analysis is implemented in Python using pandas and NumPy for data handling, Prophet for the core forecasting models, statsmodels for the SARIMA, SARIMAX and ARIMA models and the residual diagnostics, SciPy for the Box-Cox transformation, the holidays package for the German public-holiday calendar, and seaborn and matplotlib for visualisation. The primary dataset is the Open Power System Data (OPSD) daily German electricity series, obtained from [github.com/jenfly/opsd](https://github.com/jenfly/opsd) (file opsd_germany_daily.csv), a daily extract of the OPSD time series data package published at [open-power-system-data.org](https://open-power-system-data.org/), which draws on the German transmission system operators and the ENTSO-E data portal. It contains 4,383 consecutive daily observations from 1 January 2006 to 31 December 2017 in four columns, all in GWh per day: Consumption (national electricity consumption), Wind and Solar (generation) and Wind+Solar (their sum). Consumption is complete. Wind generation starts on 1 January 2010 and solar generation on 1 January 2012, with 2 and 4 isolated missing days respectively after those dates. A second dataset, the Air Passengers series used in the [ARIMA](/arima/) project, supports the bridge analysis described at the end of this section.

The workflow has seven stages:
1. Validate the data.
2. Split it chronologically into training and test periods.
3. Explore the training years, with each finding tied to a modelling decision.
4. Fit three classical baselines.
5. Fit Prophet, with hyperparameters tuned by walk-forward cross-validation.
6. Evaluate every model on the held-out test years and quantify how far the differences between models can be trusted.
7. Run a controlled experiment on wind and solar generation, then repeat the Prophet versus ARIMA comparison on the Air Passengers series.

### Data Validation

Forecasting models fail quietly on faulty inputs: a duplicated or missing date shifts every seasonal pattern that follows it, so the data is validated before any modelling. Structural checks are treated as hard failures that stop the script: the expected columns must be present, with no duplicated dates, no gaps in the daily calendar and no non-positive consumption values. All passed. Softer findings are reported rather than treated as errors: the availability windows and gap days of the wind and solar series, which matter because Prophet cannot accept a missing value in a regressor; a consistency check that Wind+Solar equals Wind plus Solar wherever all three are present (maximum discrepancy 0.0000); and a plausibility screen on the five lowest-consumption days, all of which fall in 2009, four of them Easter or Whit holidays. In a production setting these checks would sit in a formal validation suite of the kind built in the [Great Expectations](/great-expectations/) project.

### Chronological Train/Test Split

The first ten years (2006 to 2015, 3,652 days) are used for training and tuning, and the final two (2016 to 2017, 731 days) are held out for testing. A random split would be wrong for a time series, because the model would train on days on either side of the ones it is tested on and overstate its accuracy. Two test years were chosen rather than one because they contain two complete annual cycles, a more reliable verdict than a single year of weather and calendar. The split is made before exploratory analysis, so that no decision about seasonality, holidays or regressors can be influenced by the test years: the exploratory stage uses the training years only, and the test set is first used in the final evaluation, after every model has been fully specified. Within the training period, the wind and solar series define two shorter windows: 2,189 days with wind (2010 to 2015) and 1,456 days with both wind and solar (2012 to 2015), after dropping the few days with a missing value.

### Exploratory Analysis and the Decisions It Drove

Exploratory analysis here has a methodological purpose: each finding determines a specific modelling decision.

* **Level shifts**. Seasonality dominates a daily electricity series, so level changes are hard to see in the raw data. Each day is therefore compared with the same weekday 52 weeks earlier (a 364-day lag keeps the weekday aligned), which removes the seasonal pattern, so that a one-off step appears as a plateau of year-on-year change lasting about a year. Every stretch in which this change stays beyond ±5% for at least 45 days is listed, and the onset of the main step is located using regular days only, because the Christmas period distorts the comparison. This identified a persistent step of about +8.7% between 23 December 2013 and 14 January 2014, and a temporary shortfall of about 9% between May and October 2009. A candidate explanation for the step is a change in the source's coverage: the OPSD documentation describes a representativity factor for German load of 91% until 2014 and 97% since, which alone would lift reported values by about 6.6%. This is not confirmed, so the step is handled explicitly in the models, and the 2009 shortfall is carried through a sensitivity analysis (below) instead of being silently removed.
* **Seasonal profiles**. Each day is expressed relative to the mean of its own calendar year, which removes year-to-year level differences (including the step) so that only the seasonal shape remains, using regular days only.
* **Holiday effects**. The effect of a holiday is measured against what an ordinary day of the same weekday would have used, taken as the mean of the same weekday one to four weeks either side, using regular days only. This local baseline is unaffected by seasonality and by the level step. Easter Sunday and Whit Sunday are absent from the national holiday list but show clear dips against an ordinary Sunday, so they are added to the holiday calendar, and the Christmas to New Year period is split into groups of days with similar dips.
* **Additive or multiplicative seasonality**. The size of the weekly and annual swings is measured year by year and compared with the year's level. With only ten annual points the evidence is weak and mixed, so the choice is left to cross-validation.
* **Wind and solar**. Raw correlations with consumption mislead: solar peaks in summer, when demand is lowest, and both series grow with installed capacity rather than with demand. Each series and consumption are therefore first stripped of year level, weekday pattern and annual cycle (four pairs of Fourier terms), and the leftover residuals are correlated.

### Baseline Models

Three baselines of increasing sophistication are fitted, so that Prophet is compared with the best classical alternative and not with a straw man. All three forecast the full 731 test days from a single origin at the end of 2015, just as the ARIMA project forecast its 29-month holdout.
* **Seasonal naive (M0a)**. Each test day is forecast with the value from the same weekday in the most recent observed year. A 364-day lag keeps weekdays aligned; for the second forecast year the lag is 728 days, because 364 days back would fall inside the unknown test period.
* **SARIMA (M0b)**. SARIMA(1,0,1)(0,1,1)₇ combines short-term dynamics with a seasonally differenced component for the weekly cycle. By design it has no annual cycle, no holidays and no handling of the level step: it shows what a standard single-period seasonal model achieves over a two-year horizon.
* **SARIMAX (M0c)**. The same structure with regressors added: four pairs of annual Fourier terms (sine and cosine waves at the annual frequency and its first three harmonics, the standard way to represent a long seasonal cycle in an ARIMA-family model), a holiday indicator, a Christmas to New Year indicator and the level-step indicator. It therefore receives the same calendar information as Prophet, so any difference in results reflects the model and not the inputs. All of its regressors are deterministic calendar quantities, known at forecast time. SARIMAX is not covered elsewhere in this portfolio.






## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

Next steps based on current results and conclusions from above and suggested follow-up actions, analysis etc.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/Prophet_Seasonal_Forecasting_OPSD_v6.py)
