function weights = evenlyWeightDataSeries(seriesIdx)
%weights = EVENLYWIEGHTDATASERIES(seriesIdx)
% Evenly weights each data series in a data set, by weighting
% each data point relative to the number of data points in
% that data series and the total number of data series in the
% data set. This avoids the situation where short data series
% are not given as much weight. In the case of battery
% capacity fade, cells that degrade quickly often have less
% points, and we don't want this to be less important.
% Inputs:
%   obj: Instance of the ReducedOrderModel class.
%   seriesIdx (array): A column vector, denoting the data
%     series that each data point (row) belongs to across the
%     data set.
% Outputs:
%   weights (array): A column vector of weights for each data
%     point, which sum to 1.

lenDataSet = length(seriesIdx);
weights = ones(lenDataSet,1);
numSeries = length(unique(seriesIdx));
for thisSeries = unique(seriesIdx)'
    mask = seriesIdx == thisSeries;
    lenThisSeries = length(mask(mask)); % mask(mask) outputs only true elements
    weights(mask) = weights(mask).*(lenDataSet/(numSeries*lenThisSeries));
end
end