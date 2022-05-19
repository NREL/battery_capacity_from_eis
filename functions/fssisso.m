% Paul Gasper, NREL, 2020
% Work conducted under funding by Denso.

function [X_, hyp_] = fssisso(X, hyp)
%FSSISSO 
%Inputs (required):
%   X (table): feature table
%   y (double): vector of target variables
%   nNonzeroCoeffs (int): maximum number of features to select
%   nFeaturesPerSisIter (int): number of features to select per
%Inputs (optional):
%   idxSelected (double): selected feature indices
%Output:
%   X_ (table): table of selected features
%   hyp_.idxSelected (double): selected feature indices

arguments
    X table
    % Fixed hyperparameters
    hyp.y double
    hyp.nNonzeroCoeffs double
    hyp.nFeaturesPerSisIter double
    % Trained hyperparameters
    hyp.idxSelected double = []
end

% Grab data
vars = X.Properties.VariableNames;
x = X{:,:};

% Select features
if isempty(hyp.idxSelected)
    % Training
    % Create the SissoRegressor
    allL0Combinations = true;
    sisso = SissoRegressor(hyp.nNonzeroCoeffs, hyp.nFeaturesPerSisIter, allL0Combinations);
    % Run SISSO
    sisso = fitSisso(sisso, x, hyp.y);
    idxSelected = sisso.selectedIndicesL0{hyp.nNonzeroCoeffs};
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
    % Correlation of X and Y
    corrXY = abs(corr(x, y));
    ax2 = nexttile([1, 2]); hold on; box on;
    plot(corrXY, '-k', 'LineWidth', 1.5)
    % Plot chosen predictors on the line
    markerSize = 300;
    lineWidth = 2.5;
    colors = flipud(copper(length(idx)));
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
end
%}
end