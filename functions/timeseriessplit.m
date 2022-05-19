classdef timeseriessplit
    %TIMESERIESSPLIT A data splitter for time series.
    %   A data splitter for time series data designed to work as input to
    %   the RegressionPipeline testtrain method. Works with both single
    %   time series as well as a data set containing multiple independent
    %   timeseries. Data can be split by a logical mask input by the user,
    %   or by a percentage holdout. The percentage holdout assumes that all
    %   data series in the data set have been ordered, i.e., data points
    %   with larger indices occur 'after' data points with earlier indices.
    
    properties
        maskTrain
        maskTest
    end
    
    methods
        function obj = timeseriessplit(varargin)
            %TIMESERIESSPLIT Construct an instance of this class
            %   Split data according to an input logical mask or by masking
            %   off a certain percent of the data from each unique series
            %   in the data set. Assumes that the data is ordered according
            %   to time, i.e., within each data series, 
            %   Input patterns:
            %   TIMESERIESSPLIT(maskTest)
            %       maskTest (logical): vector denoting which rows in the
            %       data correspond to the test data
            %   TIMESERIESSPLIT(n, p)
            %       n (int): number of observations in the data set
            %       p (double): percentage of data held out from the end of
            %           the data set
            %   TIMESERIESSPLIT(seriesIdx, p)
            %       seriesIdx (double): vector of indices denoting which
            %           data series each row belongs to. If all entries in
            %           seriesIdx are identical (only 1 unique value), this
            %           will operate identically to the input pattern above,
            %           TIMESERIESSPLIT(length(seriesIdx), p).
            %       p (double): percentage of data held out from the end of
            %         each timeseries
            %   Output:
            %       obj
            
            if nargin == 1
                % TIMESERIESSPLIT(maskTest)
                maskTest = varargin{1};
                assert(islogical(maskTest) && isvector(maskTest),...
                    "First input 'maskTest' must be a logical vector.")
                obj.maskTest = maskTest;
                obj.maskTrain = ~maskTest;
            elseif nargin == 2
                % Input parsing
                if length(varargin{1}) == 1
                    % TIMESERIESSPLIT(n, p)
                    n = varargin{1};
                    seriesIdx = ones(n, 1);
                    p = varargin{2};
                else
                    % TIMESERIESSPLIT(seriesIdx, p)
                    seriesIdx = varargin{1};
                    p = varargin{2};
                end
                % Input validation
                assert(p>0 && p < 1, "Second input 'p' must be a double between 0 and 1 exclusive.")
                if ~iscolumn(seriesIdx)
                    seriesIdx = seriesIdx';
                end
                % Calculate the mask for each series.
                maskTest = logical(zeros(length(seriesIdx), 1));
                uniqueSeriesIdx = unique(seriesIdx, 'stable');
                for iSeries = 1:length(uniqueSeriesIdx)
                    thisSeries = uniqueSeriesIdx(iSeries);
                    maskSeries = seriesIdx == thisSeries;
                    lenSeries = length(maskSeries(maskSeries));
                    lenTestSeries = round(lenSeries*p);
                    maskTestSeries = zeros(lenSeries, 1);
                    if lenSeries > 1 & lenTestSeries > 0
                        maskTestSeries((end-(lenTestSeries-1)):end) = 1;
                    end
                    maskTestSeries = logical(maskTestSeries);
                    maskTest(maskSeries) = maskTestSeries;
                end
                obj.maskTest = maskTest;
                obj.maskTrain = ~maskTest;
            else
                error("Unrecognized input pattern.")
            end
        end
        
        function mask = training(obj)
            mask = obj.maskTrain;
        end
        
        function mask = test(obj)
            mask = obj.maskTest;
        end
    end
end

