# priority_date_prediction
Predict the priority date movement given historical data using different models

The features include last month eb2/eb3 waiting time, last month advance days, 
and month. 

Data source is from https://www.immihelp.com/visa-bulletin-tracker

Result:

It's comparably easy to predict progress/retrogress, with binary classifier, we can get AUC of ~0.8. When we use linear 
regression to predict the accurate days it's not that easy. R2 score is around 0 which means not related at all.
The days are kind of random. So I changed it to month, the r2 result looks to be the same, but AUC improves to ~0.95.

There must be some missing features, I don't have.

R2 definition: https://en.wikipedia.org/wiki/Coefficient_of_determination 