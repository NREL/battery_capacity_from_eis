classdef DummyRegressor
    %DUMMYREGRESSOR A dummy regressor.
    % Simply uses the mean of the target variable as the prediction, and
    % the standard deviation as the uncertainty.
    
    properties
        mean
        std
    end
    
    methods
        function obj = DummyRegressor(mean, std)
            %DUMMYREGRESSOR Construct an instance of this classe
            arguments
                mean double = 0
                std double = 0
            end
            obj.mean = mean;
            obj.std = std;
        end
        
        function [y, yStd, y95CI] = predict(obj, x)
            %PREDICT Makes simple predictions
            h = size(x,1);
            y = repmat(obj.mean, h, 1);
            yStd = repmat(obj.std, h, 1);
            y95CI = repmat([obj.mean - 3*obj.std, obj.mean + 3*obj.std], h, 1);
        end
    end
end

