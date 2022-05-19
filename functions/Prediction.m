classdef Prediction
    %PREDICTION Stores model inputs, predictions, and fit statistics.
    %The PREDICTION class stores model inputs and results, with
    %precalculated fit statistics to help compare models with few lines of
    %code. Includes some simple plotting methods to aid visualization.
    %Automates calculation of fit statistics/plotting for data sets with
    %multiple separate data series.
    
    properties
        % Data sorting indices
        seriesIdx
        % Indices of test folds for cross-validation
        folds
        % Feature data table
        X
        % Target data table
        Y
        % Target data prediction and uncertainty (if predicted)
        YPred
        % Residual errors
        r
        % Statistics table: r2, mae, mape, mse, rmse, msd, maxae
        FitStats
        % Statistics, separated for each unique seriesIdx
        FitStatsSeries
        % Statistics, separated for each unique cross-validation fold
        FitStatsFolds
    end
    
    methods
        function obj = Prediction(X, YPred, Y, seriesIdx, folds)
            %PREDICTION Construct an instance of this class
            %   Input data, predictions, and seriesIdx, residuals and fit
            %   statistics are then automatically calculated.
            %   Inputs (required):
            %       X (table): Input feature data
            %       YPred (table): Target data predictions
            %   Inputs (optional):
            %       Y (table): Raw target data table
            %       seriesIdx (vector): denotes unique data series within
            %           the data set.
            %       folds (vector): denotes fold number from
            %           cross-validation.
            %   Output:
            %       obj (Prediction): class instance
            
            % Input parsing
            arguments
                X table
                YPred table
                Y table = table()
                seriesIdx double = ones(height(X), 1)
                folds double = []
            end
            % Define properties
            obj.seriesIdx = seriesIdx;
            obj.folds = folds;
            obj.X = X;
            obj.Y = Y;
            obj.YPred = YPred;
            % Calculate fit metrics if there's target data to compare against
            if ~isempty(Y)
                % Calculate residuals
                obj.r = obj.Y{:,:} - obj.YPred{:,1};
                % Calculate fit statistics
                [FitStats, FitStatsSeries] = calcfitstats(obj);
                if ~isempty(folds)
                    obj_ = obj; obj_.seriesIdx = folds;
                    [~, FitStatsFolds] = calcfitstats(obj_);
                else
                    FitStatsFolds = [];
                end
            else
                FitStats = [];
                FitStatsSeries = [];
                FitStatsFolds = [];
            end
            obj.FitStats = FitStats;
            obj.FitStatsSeries = FitStatsSeries;
            obj.FitStatsFolds = FitStatsFolds;
        end
        
        %Calculate fit statistics
        function [FitStats, FitStatsSeries] = calcfitstats(obj)
            %CALFITSTATS Calculates MSD, MAE, MAPE, R2, MSE, and RMSE.
            %   Input:
            %       obj: Instance of the Prediction class.
            %   Outputs:
            %       FitStats (struct): Struct with fields corresponding to
            %           various calculated fit statistics.
            %       FitStatsSeries (struct): Array of FitStats structs,
            %           corresponding to each unique data series denoted in
            %           obj.seriesIdx.
            
            % Grab variables
            y = obj.Y{:,1};
            r = obj.r;
            uniqueSeriesIdx = unique(obj.seriesIdx, 'stable');
            
            % Mean signed difference:
            msd = sum(r)/length(r);
            % Mean absolute error:
            mae = sum(abs(r))/length(y);
            % Mean absolute percent error:
            percentError = r./y;
            percentError = percentError(~isinf(percentError) & ~isnan(percentError)); % x/y can equal NaN or Inf if y=0
            mape = sum(abs(percentError))/length(y);
            % Coefficient of determination:
            r2 = 1 - sum(r.^2)./sum((y - mean(y)).^2);
            % Mean squared error:
            mse = sum(r.^2)/length(y);
            % Root mean squared error:
            rmse = sqrt(mse);
            % Max absolute error:
            maxae = max(abs(r));
            
            % Create FitStats struct:
            FitStats = table(mae, mape, mse, rmse, r2, msd, maxae);
            
            % Repeat for each unique series
            fitStatsSeries = zeros(length(uniqueSeriesIdx), 8);
            for iSeries = 1:length(uniqueSeriesIdx)
                % Get data from this series
                thisSeries = uniqueSeriesIdx(iSeries);
                maskThisSeries = obj.seriesIdx == thisSeries;
                y_ = y(maskThisSeries);
                r_ = r(maskThisSeries);
                
                % Calculate stats
                % Mean signed difference:
                msd = sum(r_)/length(r_);
                % Mean absolute error:
                mae = sum(abs(r_))/length(y_);
                % Mean absolute percent error:
                percentError = r_./y_;
                percentError = percentError(~isinf(percentError) & ~isnan(percentError)); % x/y_ can equal NaN or Inf if y_=0
                mape = sum(abs(percentError))/length(y_);
                % Coefficient of determination:
                r2 = 1 - sum(r_.^2)./sum((y_ - mean(y_)).^2);
                % Mean squared error:
                mse = sum(r_.^2)/length(y_);
                % Root mean squared error:
                rmse = sqrt(mse);
                % Max absolute error:
                maxae = max(abs(r_));
                
                % Store results
                fitStatsSeries(iSeries, :) = [thisSeries, mae, mape, mse, rmse, r2, msd, maxae];
            end
            FitStatsSeries = array2table(fitStatsSeries, 'VariableNames', {'seriesIdx','mae','mape','mse','rmse','r2','msd','maxae'});
        end
        
        %Plotting
        %plotyy(actual v predicted) (enable color by seriesIdx, legend entries by seriesIdx)
        function plotyy(obj, markerStyle, plotOpt, lineOpt)
            arguments
                obj Prediction
                markerStyle char = '.'
                plotOpt.ax = []
                plotOpt.SeriesLineStyle char = []
                plotOpt.SeriesColors double = []
                plotOpt.Legend char = 'off'
                plotOpt.LegendEntries = []
                plotOpt.LegendLocation char = 'northwest'
                plotOpt.ConfidenceIntervalAlpha double = []
                plotOpt.YEqualsXLine char = 'on'
                lineOpt.?matlab.graphics.chart.primitive.Line
            end
            numSeries = length(unique(obj.seriesIdx));
            if ~isempty(plotOpt.SeriesColors)
                assert(all(size(plotOpt.SeriesColors) == [numSeries, 3]),...
                    "Must have one color for each line.")
            end
            if ~isempty(plotOpt.ConfidenceIntervalAlpha)
                assert(plotOpt.ConfidenceIntervalAlpha < 1 && plotOpt.ConfidenceIntervalAlpha > 0,...
                    "ConfidenceIntervalAlpha must be a percentage less than 1.")
            end
            lineOpt = namedargs2cell(lineOpt);
            
            if isempty(plotOpt.ax)
                figure; hold on; box on; grid on;
            end
            if strcmpi(plotOpt.SeriesLineStyle, 'off') && isempty(plotOpt.SeriesColors)
                if ~isempty(plotOpt.ConfidenceIntervalAlpha)
                    err = obj.YPred{:,2} .* abs(norminv((1-plotOpt.ConfidenceIntervalAlpha)/2));
                    if ~isempty(plotOpt.ax)
                        errorbar(plotOpt.ax, obj.Y{:,:}, obj.YPred{:,1}, err, markerStyle, lineOpt{:});
                    else
                        errorbar(obj.Y{:,:}, obj.YPred{:,1}, err, markerStyle, lineOpt{:});
                    end
                else
                    if ~isempty(plotOpt.ax)
                        plot(plotOpt.ax, obj.Y{:,:}, obj.YPred{:,1}, markerStyle, lineOpt{:})
                    else
                        plot(obj.Y{:,:}, obj.YPred{:,1}, markerStyle, lineOpt{:})
                    end
                end
            else 
                uniqueSeries = unique(obj.seriesIdx, 'stable');
                lineStyle = [plotOpt.SeriesLineStyle, markerStyle];
                go = gobjects(numSeries,1);
                for i = 1:length(uniqueSeries)
                    thisSeries = uniqueSeries(i);
                    maskSeries = obj.seriesIdx == thisSeries;
                    if ~isempty(plotOpt.ConfidenceIntervalAlpha)
                        err = obj.YPred{maskSeries,2} .* abs(norminv((1-plotOpt.ConfidenceIntervalAlpha)/2));
                        if ~isempty(plotOpt.SeriesColors)
                            if ~isempty(plotOpt.ax)
                                go(i) = errorbar(plotOpt.ax, obj.Y{maskSeries,:}, obj.YPred{maskSeries,1}, err, lineStyle, 'Color', plotOpt.SeriesColors(i,:), lineOpt{:});
                            else
                                go(i) = errorbar(obj.Y{maskSeries,:}, obj.YPred{maskSeries,1}, err, lineStyle, 'Color', plotOpt.SeriesColors(i,:), lineOpt{:});
                            end
                        else
                            if ~isempty(plotOpt.ax)
                                go(i) = errorbar(plotOpt.ax, obj.Y{maskSeries,:}, obj.YPred{maskSeries,1}, err, lineStyle, lineOpt{:});
                            else
                                go(i) = errorbar(obj.Y{maskSeries,:}, obj.YPred{maskSeries,1}, err, lineStyle, lineOpt{:});
                            end
                        end
                    end
                    if ~isempty(plotOpt.SeriesColors)
                        if ~isempty(plotOpt.ax)
                            go(i) = plot(plotOpt.ax, obj.Y{maskSeries,:}, obj.YPred{maskSeries,1}, lineStyle, 'Color', plotOpt.SeriesColors(i,:), lineOpt{:});
                        else
                            go(i) = plot(obj.Y{maskSeries,:}, obj.YPred{maskSeries,1}, lineStyle, 'Color', plotOpt.SeriesColors(i,:), lineOpt{:});
                        end
                    else
                        if ~isempty(plotOpt.ax)
                            go(i) = plot(plotOpt.ax, obj.Y{maskSeries,:}, obj.YPred{maskSeries,1}, lineStyle, lineOpt{:});
                        else
                            go(i) = plot(obj.Y{maskSeries,:}, obj.YPred{maskSeries,1}, lineStyle, lineOpt{:});
                        end
                    end
                end
            end
            if strcmpi(plotOpt.YEqualsXLine, 'on')
                axlims = axis;
                maxlim = max(abs(axlims));
                if ~isempty(plotOpt.ax)
                    plot(plotOpt.ax, [-maxlim, maxlim], [-maxlim, maxlim], '--k', 'LineWidth', 1.5)
                else
                    plot([-maxlim, maxlim], [-maxlim, maxlim], '--k', 'LineWidth', 1.5)
                end
                axis(axlims);
            end
            if strcmpi(plotOpt.Legend, 'on') && exist('go','var')
                if ~isempty(plotOpt.LegendEntries)
                    legend(go, plotOpt.LegendEntries, 'Location', plotOpt.LegendLocation);
                else
                    legend(go, 'Location', plotOpt.LegendLocation)
                end
            end
        end
        
        %histr(residuals histogram) (enable color/separation by seriesIdx, legend entries by seriesIdx)
        %plotxy(x v actual, x v predicted) (enable color by seriesIdx, choose x var, legend entries by seriesIdx, allow xvar to be input as column table)
        %plotxr(x v residuals) (enable color by seriesIdx, choose x var, legend entries by seriesIdx, allow xvar to be input as column table)
    end
end

