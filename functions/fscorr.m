% Paul Gasper, NREL, 2020

function [X_, hyp_] = fscorr(X, hyp)
%FSCORR Sequentially selects n features (columns of X) that are highly
%correlated with Y but not correlated with one another. At each step, the
%highest correlated feature of X is selected (X*); then, the correlations
%between X* and all other remaining features in X are calculated, and
%features with correlations above the percentile threshold p are removed
%from X, so that redundant features are not chosen on the next iteration.
%To my knowledge, this feature selection algorithm was first proposed by
%Greenback and Howey (DOI 10.1109/TII.2021.3106593).
%Inputs (required):
%   X (table): feature table
%   hyp.y (double): target variable vector
%   hyp.n (int): number of features to select
%   hyp.p (0 to 1): percent cutoff for removing redundant features
%Inputs (optional):
%   hyp.idxSelected (double): selected feature indicies
%Output:
%   X_ (table): table with selected features
%   hyp_.idxSelected (double): array of selected feature indicies

arguments
    X table
    % Fixed hyperparameters
    hyp.y double
    hyp.n double
    hyp.p double
    % Trained hyperparameters
    hyp.idxSelected double = []
end

% Grab data
vars = X.Properties.VariableNames;
x = X{:,:};

% Select features
if isempty(hyp.idxSelected)
    % Training
    y = hyp.y;
    n = hyp.n;
    p = hyp.p * 100;
    % Select features
    idxSelected = zeros(1,n);
    idxRemaining = 1:size(x,2);
    for iFeature = 1:n
        % Find correlations between remaining x and y:
        xRemaining = x(:,idxRemaining);
        corrXY = abs(corr(xRemaining, y));
        % Find the maximum correlated feature:
        [~, idxFeature] = max(corrXY);
        idxSelected(iFeature) = idxRemaining(idxFeature);
        % Remove the selected feature from xRemaining
        xSelected = xRemaining(:,idxFeature);
        idxRemaining(idxFeature) = [];
        xRemaining(:,idxFeature) = [];
        % Remove any features highly correlated to the feature just extracted.
        % In this case, highly correlated is anything that is above the 'p'th
        % percentile of correlations between the feature just selected and the
        % remaining features.
        corrXX = abs(corr(xSelected, xRemaining));
        maskKeep = corrXX < prctile(corrXX, p);
        idxRemaining = idxRemaining(maskKeep);
        if numel(idxRemaining) <= n - iFeature
            warning("Not enough predictors left, use a higher percentile cutoff (p) or fewer features (n).")
            idxSelected = idxSelected(1:iFeature);
            break;
        end
    end
else
    % Testing
    idxSelected = hyp.idxSelected;
end

hyp_ = {'idxSelected', idxSelected};
x_ = x(:, idxSelected);
vars = vars(idxSelected);

% Return
X_ = array2table(x_, 'VariableNames', vars);

%{
if flagPlot
    figure; t = tiledlayout('flow', 'TileSpacing', 'compact');
    % Correlation plot of XX (contourf gives more control than heatmap)
    corrXX = abs(corr(x));
    ax1 = nexttile([2, 2]); hold on; 
    [~,c] = contourf(corrXX);
    colormap(brewermap('Blues', 256));
    ax1.CLim = [0 1];
    set(ax1, 'YDir', 'reverse')
    cb = colorbar;
    cb.Label.String = '|corr(X)|';
    % Plot chosen predictors on the diagonal
    markerSize = 300;
    lineWidth = 2.5;
    colors = flipud(copper(length(idx)));
    for iPredictor = 1:length(idx)
        value = idx(iPredictor);
        scatter(value, value, markerSize, colors(iPredictor,:), 'p', 'filled')
        xline(value, '--', 'LineWidth', lineWidth, 'Color', colors(iPredictor,:));
        yline(value, '--', 'LineWidth', lineWidth, 'Color', colors(iPredictor,:));
        % reduce marker size and linewidth for each predictor:
        markerSize = markerSize * 0.7;
        lineWidth = lineWidth * 0.85;
    end
    axis('square');
    ylabel('Predictor Index');
    % Correlation of X and Y
    corrXY = abs(corr(x, y));
    ax2 = nexttile([1, 2]); hold on; box on;
    plot(corrXY, '-k', 'LineWidth', 1.5)
    % Plot chosen predictors on the line
    markerSize = 300;
    lineWidth = 2.5;
    for iPredictor = 1:length(idx)
        value = idx(iPredictor);
        scatter(value, corrXY(value), markerSize, colors(iPredictor,:), 'p', 'filled')
        xline(value, '--', 'LineWidth', lineWidth, 'Color', colors(iPredictor,:));
        % reduce marker size and linewidth for each predictor:
        markerSize = markerSize * 0.7;
        lineWidth = lineWidth * 0.85;
    end
    xlim([1 length(corrXY)])
    ylim([0 1])
    ylabel('|corr(X, Y)|');
    xlabel('Predictor Index');
    % Set figure position
    set(gcf, 'Units', 'inches', 'Position', [1.979166666666667,2.822916666666667,4.875,6.6875])
end
%}
end