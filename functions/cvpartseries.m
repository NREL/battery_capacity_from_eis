classdef cvpartseries
    %CVPARTSERIES Like cvpartition, but makes splits by data series
    %   CVPARTSERIES splits data using the same arguments as cvparition,
    %   but splits data along the basis of data series rather than data
    %   point, which is useful for certain applications.
    
    properties
        cv
        seriesIdx
        uniqueSeriesIdx
        NumTestSets
        NumObservations
        TestSize
        TrainSize
        Type
    end
    
    methods
        function obj = cvpartseries(seriesIdx, type, k)
            %CVPARTSERIES Construct an instance of this class
            %   Constructs an object that splits data according to a unique
            %   value of the input seriesIdx, rather than by data point.
            %   Inputs (required):
            %       seriesIdx (double): vector of series indices, denoting
            %           unique series each data point corresponds to. The
            %           number of observations to split by is the number of
            %           unique values in seriesIdx.
            %       type (char): type of split, either 'KFold', 'Holdout',
            %           or 'Leaveout'.
            %   Inputs (optional):
            %       k (double): either the number of folds in KFold, or the
            %           percentage held out for testing in Holdout.
            %   Output:
            %       obj
            
            % Input parsing
            arguments
                seriesIdx double
                type char {mustBeMember(type, {'KFold', 'Holdout', 'Leaveout'})}
                k double = []
                
            end
            
            % Input validation
            if any(strcmp(type, {'KFold', 'Holdout'}))
                assert(nargin == 3, "Must provide a value for input k (# of folds or % held out for test).")
            end
            if ~iscolumn(seriesIdx)
                seriesIdx = seriesIdx';
            end
            
            % Generate properties
            uniqueSeriesIdx = unique(seriesIdx, 'stable');
            n = length(uniqueSeriesIdx);
            switch type
                case {'KFold', 'Holdout'}
                    cv = cvpartition(n, type, k);
                case {'Leaveout'}
                    cv = cvpartition(n, type);
            end
            % Calculate train/test size
            TrainSize = zeros(1, cv.NumTestSets);
            TestSize = zeros(1, cv.NumTestSets);
            for ifold = 1:cv.NumTestSets
                trainSeries = uniqueSeriesIdx(training(cv, ifold));
                maskTrain = any(seriesIdx == trainSeries', 2);
                TrainSize(ifold) = length(maskTrain(maskTrain));
                testSeries = uniqueSeriesIdx(test(cv, ifold));
                maskTest = any(seriesIdx == testSeries', 2);
                TestSize(ifold) = length(maskTest(maskTest));
            end
            
            % Assign properties
            obj.cv = cv;
            obj.seriesIdx = seriesIdx;
            obj.uniqueSeriesIdx = uniqueSeriesIdx;
            obj.NumTestSets = cv.NumTestSets;
            obj.NumObservations = length(seriesIdx);
            obj.TestSize = TestSize;
            obj.TrainSize = TrainSize;
            obj.Type = type;
        end
        
        function mask = training(obj, i)
            %TRAINING Output training mask for fold i
            %   Inputs (required):
            %       obj
            %   Inputs (optional):
            %       i (int): fold number, required if the cv type is
            %       'KFold' or 'Leaveout'
            %   Output:
            %       mask (logical): indexing vector
            
            % Input parsing
            arguments
                obj
                i double = []
            end
            % Input validation
            if any(strcmp(obj.Type, {'KFold', 'Leaveout'}))
                assert(nargin == 2, "Must input the fold number, i, for 'KFold' or 'Leaveout' types.")
            end
            
            trainSeries = obj.uniqueSeriesIdx(training(obj.cv, i));
            mask = any(obj.seriesIdx == trainSeries', 2);
        end
        
        function mask = test(obj, i)
            %TEST Output test mask for fold i
            %   Inputs (required):
            %       obj
            %   Inputs (optional):
            %       i (int): fold number, required if the cv type is
            %       'KFold' or 'Leaveout'
            %   Output:
            %       mask (logical): indexing vector
            
            % Input parsing
            arguments
                obj
                i double = []
            end
            % Input validation
            if any(strcmp(obj.Type, {'KFold', 'Leaveout'}))
                assert(nargin == 2, "Must input the fold number, i, for 'KFold' or 'Leaveout' types.")
            end
            
            testSeries = obj.uniqueSeriesIdx(test(obj.cv, i));
            mask = any(obj.seriesIdx == testSeries', 2);
        end
        
        % reparition function
    end
end

