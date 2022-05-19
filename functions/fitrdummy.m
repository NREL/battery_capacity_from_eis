function out = fitrdummy(x, y)
%FITRDUMMY Creates a DummyRegressor on this data.
out = DummyRegressor(mean(y), std(y));
end

