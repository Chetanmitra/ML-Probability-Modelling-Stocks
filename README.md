# ML-Probability-Modelling-Stocks
Prediction on future stock data based on selected data. Uses PRNG and historic data to estimate and plot the data.

This is just surface level as it is missing numerous potential mathmetical equations. With greater data (and a more powerful computer), it would be much quicker to train and run this model. 

How the program runs currently:
1. Call a ticker and set the start/end and period for each data point 
(Picking the right time frame matters a lot as taking a large time frame for daily will give a mostly upwards directional chart and ignore volatility. SPY has mostly gone up the last 10 years, with downtrends lasting for significantly less time)
3. Gather percentage change for each data point (Forward Close - Previous Close)
4. Group the data by percentage change (Currently its 0-0.5,0.5-1,1+ and same for negative zones)
5. Collect data for the change next period
6. Select a random point from the next period, set it as the last percentage value and calculate the next random closing point
7. Currently set to predict the future 100 periods but that is flexible 
8. Run this for 10 instance (Also flexible) and then calculate the average of each x value to reduce outliers 
(Averaging too many instances will create a linearized data which ignores volatility. Keep the number low to bring down/up outliers accordingly)
10. Plot it numerous times on the same chart

Future Plans:
1. Check for Outlier on the printed Data
2. Add more statistical function to refine the charts.

100 projections for 100 hours Image v1:

![image](https://user-images.githubusercontent.com/70171751/193443348-6bc461b8-6486-4bc7-85bb-256c424b2980.png)

You can see the data is extremely scattered here. However, the data is projecting maximum highs of 370 and maximum lows of 345 for the next 100 hours.

100 projections for 100 days Image v1:

![image](https://user-images.githubusercontent.com/70171751/193443711-f585df2f-698f-4590-a33a-40e681f3052f.png)

Similar to the hourly data, there is abnormal projections with market moving straight sideways. However the projected lows for next 100 days is 310 with highs being 390.
