% Paul Gasper, NREL, 5/2022
%{
Replicates some of the more interesting models explored in the work.
%}

clear; close all; clc;
addpath(genpath('functions'))

% Load the data tables
load('data\Data_Denso2021.mat', 'Data', 'DataFormatted', 'Data2Formatted')
freq = Data.Freq(1,:);
% Filter out noisy interpolated data
idxKeep = filterInterpData(Data);
DataFormatted = DataFormatted(idxKeep, :);
Data = combineDataTables(DataFormatted, Data2Formatted);
% Remove BOL data (seriesIdx = 43 & 44, the magnitude of the impedance is
% quite a bit different for these cells compared to the aging test matrix cells).
Data = Data(1:1194, :);
clearvars DataFormatted Data2Formatted

% Some colors for plots
colors1 = brewermap('RdPu', 22);
colors1 = colors1(end-10:end,:);
colors2 = brewermap('Blues', 38);
colors2 = colors2(end-17:end,:);
colors3 = brewermap('Greens', 5);
colors3 = colors3(end-1:end,:);
colortriplet = [colors1(3,:); colors2(12,:); colors3(2,:)];
cellsTest = [7,10,13,17,24,30,31];
colors = [colors1;colors2;colors3];
idxTest = any([1:31]' == cellsTest, 2);
colorsTrain = colors(~idxTest, :);
colorsTest = colors(idxTest, :);

clearvars -except Data freq colortriplet colorsTrain colorsTest

%% Search for optimal single frequency model, linear estimator, -10C EIS
% Mask off just -10C EIS data
mask_m10C = Data.TdegC_EIS == -10 & Data.soc_EIS == 0.5;
Data_m10C = Data(mask_m10C, :);
% Pull out X and Y data tables. X variables are any Zreal, Zimag, Zmag, and
% Zphz data points. Y variable is the relative discharge capacity, q.
X = Data_m10C(:, 6:end); Y = Data_m10C(:,2); seriesIdx = Data_m10C{:, 1};
% Data from cells running a WLTP drive cycle, and some from the aging study
% are used as test data.
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data_m10C.seriesIdx == cellsTest,2);
data.Xtest = X(maskTest,:); data.Ytest = Y(maskTest,:); 
seriesIdxTest = seriesIdx(maskTest);
% Cross-validation and training are conducted on the same data split.
data.Xcv = X(~maskTest,:); data.Ycv = Y(~maskTest,:); 
seriesIdxCV = seriesIdx(~maskTest);
% Define a cross-validation scheme.
cvsplit = cvpartseries(seriesIdxCV, 'Leaveout');

% Define the estimator
estimator = @fitlm;

% Define model pipelines
name = strings(length(freq), 1);
idxFreq = [1:length(freq)]';
for i = 1:length(freq)
    seq = {@RegressionPipeline.normalizeZScore,...
        @selectFrequency};
    hyp = {{}, {"idxFreq", idxFreq(i)}};
    Pipe = RegressionPipeline(estimator,...
        "FeatureTransformationSequence", seq,...
        "FeatureTransformationFixedHyp", hyp);
    name(i) = "Model_1B_" + i;
    if strcmp(func2str(estimator), 'fitrensemble')
        Pipe.ModelFuncOpts = {'Method','bag'};
    end
    Model(i) = Pipe;
end
Model = transpose(Model);
Pipes = table(name, idxFreq, Model);

% Fit pipelines
Pipes = fitPipes(Pipes, data, cvsplit, seriesIdxCV, seriesIdxTest);
clearvars -except Data freq colortriplet colorsTrain colorsTest Pipes

% Plot MAE vs. frequency
figure; hold on; box on; grid on;
plot(freq, Pipes.maeCrossVal, ':', 'Color', colortriplet(1,:), 'LineWidth', 1.5, 'AlignVertexCenters', 'on')
plot(freq, Pipes.maeTest,     '-', 'Color', colortriplet(1,:), 'LineWidth', 1.5)
% decorations
legend("Linear, CV", "Linear, Test")
set(gca, 'XScale', 'log'); ylim([0 Inf])
xlabel('Frequency (Hz)');
ylabel('MAE (-10\circC EIS)')
set(gcf, 'Units', 'inches', 'Position', [3.5,5,3.25,2.5])

%% GPR double frequency model trained on all EIS data
% Use the optimal frequencies found by the exhaustive search.

% Grab X and Y data. Same train/cv/test splits as above.
X = Data(:, 6:end); Y = Data(:,2); seriesIdx = Data{:, 1};
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data.seriesIdx == cellsTest,2);
data.Xtest = X(maskTest,:);  data.Ytest = Y(maskTest,:);  seriesIdxTest = seriesIdx(maskTest);
data.Xcv   = X(~maskTest,:); data.Ycv   = Y(~maskTest,:); seriesIdxCV   = seriesIdx(~maskTest);
cvsplit = cvpartseries(seriesIdxCV, 'Leaveout');

% Define the pipeline
seq = {@RegressionPipeline.normalizeZScore,...
    @selectFrequency};
hyp = {{}, {"idxFreq", [22,35]}};
Pipe_GPR = RegressionPipeline(@fitrgp,...
    "FeatureTransformationSequence", seq,...
    "FeatureTransformationFixedHyp", hyp);
% Train and cross-validate
[Pipe_GPR, PTrainGPR, ~, PCrossValGPR] = crossvalidate(Pipe_GPR, data.Xcv, data.Ycv, cvsplit, seriesIdxCV);
% Test
PTestGPR = predict(Pipe_GPR, data.Xtest, data.Ytest, seriesIdxTest);

% 1D partial dependence of all features
features = Pipe_GPR.FeatureVars;
figure; tiledlayout('flow');
ax = [];
for i = 1:length(features)
    ax_ = nexttile; box on;
    plotPartialDependence(Pipe_GPR.Model, i);
    title(features(i), 'FontWeight', 'normal', 'Interpreter', 'none')
    ax = [ax,ax_];
end
linkaxes(ax, 'y')

% 2D partial dependence plot of the two most impactful features
idxFeatures = [1,8]; features = Pipe_GPR.FeatureVars;
[pd, x, y] = partialDependence(Pipe_GPR.Model, idxFeatures);
% 2D plot
figure;
contourf(x, y, pd); colormap(plasma(256)); hold on; grid on; box on;
% scatter of feature values
scatter(PTrainGPR.X{:,idxFeatures(1)}, PTrainGPR.X{:,idxFeatures(2)}, 40, '.k')
% decorations
cb = colorbar(); cb.Label.String = "Rel. discharge capacity";
xlabel(strrep(features(idxFeatures(1)),'_',' '));
ylabel(strrep(features(idxFeatures(2)),'_',' '));
axis square;
set(gcf, 'Units', 'inches', 'Position', [8,2,3.25,2.5]);

% yy plot
figure; t=tiledlayout(1,3,'Padding', 'compact', 'TileSpacing', 'compact');
ax1 = nexttile; hold on; box on; grid on;
plotyy(PTrainGPR, 'k.', 'ax', ax1, 'SeriesColors', colorsTrain, 'MarkerSize', 8)
xlabel('Actual rel. capacity'); 
ylabel("Predicted rel. capacity");
% axis square; axis equal
ax2 = nexttile; hold on; box on; grid on;
plotyy(PCrossValGPR, 'k.', 'ax', ax2, 'SeriesColors', colorsTrain, 'MarkerSize', 8)
xlabel('Actual rel. discharge capacity'); 
ylabel("Predicted rel. capacity");
% axis square; axis equal
ax3 = nexttile; hold on; box on; grid on;
plotyy(PTestGPR, 'k.', 'ax', ax3, 'SeriesColors', colorsTest, 'MarkerSize', 8)
xlabel('Actual rel. discharge capacity'); 
ylabel("Predicted rel. capacity");
% axis square; axis equal
linkaxes([ax1, ax2, ax3])
set(gcf, 'Units', 'inches', 'Position', [3.5,5,6.5,2])
annotation(gcf,'textbox',...
    [0.079525641025641 0.807291666666667 0.045474358974359 0.125],...
    'String',{'a'},...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.400038461538461 0.807291666666667 0.0454743589743589 0.125],'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.715743589743589 0.807291666666667 0.0454743589743589 0.125],'String','c',...
    'FitBoxToText','off',...
    'EdgeColor','none');

clearvars -except Data freq colortriplet colorsTrain colorsTest Pipes Pipe_GPR PTestGPR

%% RF double frequency model trained on all EIS data
% Use the optimal frequencies found by the exhaustive search.

% Grab X and Y data. Same train/cv/test splits as above.
X = Data(:, 6:end); Y = Data(:,2); seriesIdx = Data{:, 1};
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data.seriesIdx == cellsTest,2);
data.Xtest = X(maskTest,:);  data.Ytest = Y(maskTest,:);  seriesIdxTest = seriesIdx(maskTest);
data.Xcv   = X(~maskTest,:); data.Ycv   = Y(~maskTest,:); seriesIdxCV   = seriesIdx(~maskTest);
cvsplit = cvpartseries(seriesIdxCV, 'Leaveout');

% RandomForest
estimator = @fitrensemble;
seq = {@RegressionPipeline.normalizeZScore,...
    @selectFrequency};
hyp = {{}, {"idxFreq", [18,34]}};
Pipe_RF = RegressionPipeline(estimator,...
    "FeatureTransformationSequence", seq,...
    "FeatureTransformationFixedHyp", hyp,...
    "ModelFuncOpts", {'Method','bag'});
[Pipe_RF, PTrainRF, ~, PCrossValRF] = crossvalidate(Pipe_RF, data.Xcv, data.Ycv, cvsplit, seriesIdxCV);
PTestRF = predict(Pipe_RF, data.Xtest, data.Ytest, seriesIdxTest);

% 1D partial dependence of all features
features = Pipe_RF.FeatureVars;
figure; tiledlayout('flow');
ax = [];
for i = 1:length(features)
    ax_ = nexttile; box on;
    plotPartialDependence(Pipe_RF.Model, i);
    title(features(i), 'FontWeight', 'normal', 'Interpreter', 'none')
    ax = [ax,ax_];
end
linkaxes(ax, 'y')

% 2D partial dependence plot of the two most impactful features
idxFeatures = [8,1]; features = Pipe_RF.FeatureVars;
[pd, x, y] = partialDependence(Pipe_RF.Model, idxFeatures);
% 2D plot
figure;
contourf(x, y, pd); colormap(plasma(256)); hold on; grid on; box on;
% scatter of feature values
scatter(PTrainRF.X{:,idxFeatures(1)}, PTrainRF.X{:,idxFeatures(2)}, 40, '.k')
% decorations
cb = colorbar(); cb.Label.String = "Rel. discharge capacity";
xlabel(strrep(features(idxFeatures(1)),'_',' '));
ylabel(strrep(features(idxFeatures(2)),'_',' '));
axis square;
set(gcf, 'Units', 'inches', 'Position', [8,2,3.25,2.5]);

% yy plot
figure; t=tiledlayout(1,3,'Padding', 'compact', 'TileSpacing', 'compact');
ax1 = nexttile; hold on; box on; grid on;
plotyy(PTrainRF, 'k.', 'ax', ax1, 'SeriesColors', colorsTrain, 'MarkerSize', 8)
xlabel('Actual rel. capacity'); 
ylabel("Predicted rel. capacity");
% axis square; axis equal
ax2 = nexttile; hold on; box on; grid on;
plotyy(PCrossValRF, 'k.', 'ax', ax2, 'SeriesColors', colorsTrain, 'MarkerSize', 8)
xlabel('Actual rel. discharge capacity'); 
ylabel("Predicted rel. capacity");
% axis square; axis equal
ax3 = nexttile; hold on; box on; grid on;
plotyy(PTestRF, 'k.', 'ax', ax3, 'SeriesColors', colorsTest, 'MarkerSize', 8)
xlabel('Actual rel. discharge capacity'); 
ylabel("Predicted rel. capacity");
% axis square; axis equal
linkaxes([ax1, ax2, ax3])
set(gcf, 'Units', 'inches', 'Position', [3.5,5,6.5,2])
annotation(gcf,'textbox',...
    [0.079525641025641 0.807291666666667 0.045474358974359 0.125],...
    'String',{'a'},...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.400038461538461 0.807291666666667 0.0454743589743589 0.125],'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.715743589743589 0.807291666666667 0.0454743589743589 0.125],'String','c',...
    'FitBoxToText','off',...
    'EdgeColor','none');

clearvars -except Data freq colortriplet colorsTrain colorsTest Pipes Pipe_GPR PTestGPR Pipe_RF PTestRF

%% RF double frequency model trained on all EIS data with weights
% Use the optimal frequencies found by the exhaustive search. Weight the
% regression to prioritize balancing error across all 4 EIS temperatures.

% Grab X and Y data. Same train/cv/test splits as above.
X = Data(:, 6:end); Y = Data(:,2); seriesIdx = Data{:, 1};
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data.seriesIdx == cellsTest,2);
data.Xtest = X(maskTest,:);  data.Ytest = Y(maskTest,:);  seriesIdxTest = seriesIdx(maskTest);
data.Xcv   = X(~maskTest,:); data.Ycv   = Y(~maskTest,:); seriesIdxCV   = seriesIdx(~maskTest);
cvsplit = cvpartseries(seriesIdxCV, 'Leaveout');

% Calculate weights
weights = evenlyWeightDataSeries(Data.TdegC_EIS(~maskTest));

% RandomForest
estimator = @fitrensemble;
seq = {@RegressionPipeline.normalizeZScore,...
    @selectFrequency};
hyp = {{}, {"idxFreq", [33,43]}};
ModelOpts.Weights = weights;
Pipe_RF_2 = RegressionPipeline(estimator,...
    "FeatureTransformationSequence", seq,...
    "FeatureTransformationFixedHyp", hyp,...
    "ModelOpts", ModelOpts,...
    "ModelFuncOpts", {'Method','bag'});
[Pipe_RF_2, PTrainRF_2, ~, PCrossValRF_2] = crossvalidate(Pipe_RF_2, data.Xcv, data.Ycv, cvsplit, seriesIdxCV);
PTestRF_2 = predict(Pipe_RF_2, data.Xtest, data.Ytest, seriesIdxTest);

% 1D partial dependence of all features
features = Pipe_RF_2.FeatureVars;
figure; tiledlayout('flow');
ax = [];
for i = 1:length(features)
    ax_ = nexttile; box on;
    plotPartialDependence(Pipe_RF_2.Model, i);
    title(features(i), 'FontWeight', 'normal', 'Interpreter', 'none')
    ax = [ax,ax_];
end
linkaxes(ax, 'y')

% 2D partial dependence plot of the two most impactful features
idxFeatures = [3,8]; features = Pipe_RF_2.FeatureVars; %[3,8] or [7,8]
[pd, x, y] = partialDependence(Pipe_RF_2.Model, idxFeatures);
% 2D plot
figure;
contourf(x, y, pd); colormap(plasma(256)); hold on; grid on; box on;
% scatter of feature values
scatter(PTrainRF_2.X{:,idxFeatures(1)}, PTrainRF_2.X{:,idxFeatures(2)}, 40, '.k')
% decorations
cb = colorbar(); cb.Label.String = "Rel. discharge capacity";
xlabel(strrep(features(idxFeatures(1)),'_',' '));
ylabel(strrep(features(idxFeatures(2)),'_',' '));
axis square;
set(gcf, 'Units', 'inches', 'Position', [8,2,3.25,2.5]);

% yy plot
figure; t=tiledlayout(1,3,'Padding', 'compact', 'TileSpacing', 'compact');
ax1 = nexttile; hold on; box on; grid on;
plotyy(PTrainRF_2, 'k.', 'ax', ax1, 'SeriesColors', colorsTrain, 'MarkerSize', 8)
xlabel('Actual rel. capacity'); 
ylabel("Predicted rel. capacity");
% axis square; axis equal
ax2 = nexttile; hold on; box on; grid on;
plotyy(PCrossValRF_2, 'k.', 'ax', ax2, 'SeriesColors', colorsTrain, 'MarkerSize', 8)
xlabel('Actual rel. discharge capacity'); 
ylabel("Predicted rel. capacity");
% axis square; axis equal
ax3 = nexttile; hold on; box on; grid on;
plotyy(PTestRF_2, 'k.', 'ax', ax3, 'SeriesColors', colorsTest, 'MarkerSize', 8)
xlabel('Actual rel. discharge capacity'); 
ylabel("Predicted rel. capacity");
% axis square; axis equal
linkaxes([ax1, ax2, ax3])
set(gcf, 'Units', 'inches', 'Position', [3.5,5,6.5,2])
annotation(gcf,'textbox',...
    [0.079525641025641 0.807291666666667 0.045474358974359 0.125],...
    'String',{'a'},...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.400038461538461 0.807291666666667 0.0454743589743589 0.125],'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.715743589743589 0.807291666666667 0.0454743589743589 0.125],'String','c',...
    'FitBoxToText','off',...
    'EdgeColor','none');

clearvars -except Data freq colortriplet colorsTrain colorsTest Pipes Pipe_GPR PTestGPR Pipe_RF PTestRF Pipe_RF_2 PTestRF_2

%% Compare unweighted GPR and weighted RF models
figure; t=tiledlayout(1,4,'Padding','compact','TileSpacing','compact');
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data.seriesIdx == cellsTest,2);
T = Data.TdegC_EIS(maskTest);
q = linspace(0.7, 1.05);
lw = 1.5;
% unweighted
nexttile; hold on; box on;
pd = fitdist(PTestGPR.Y{:,:}, 'kernel');
plot(q, pdf(pd,q), '-k', 'LineWidth', lw);
pd = fitdist(PTestGPR.YPred{:,1}, 'kernel');
plot(q, pdf(pd,q), '-', 'Color', colortriplet(1,:), 'LineWidth', lw);
pd = fitdist(PTestRF.YPred{:,1}, 'kernel');
plot(q, pdf(pd,q), '-', 'Color', colortriplet(2,:), 'LineWidth', lw);
pd = fitdist(PTestRF_2.YPred{:,1}, 'kernel');
plot(q, pdf(pd,q), '-', 'Color', colortriplet(3,:), 'LineWidth', lw);
xlabel('Rel. capacity');
title('All data', 'FontWeight', 'normal')
lgd = legend('Actual', 'GPR (unweighted)', 'RF (unweighted)', 'RF (weighted)', 'NumColumns', 4); lgd.Layout.Tile = 'north';
% by temp
temps = unique(T);
for i = 1:length(temps)
    temperature = temps(i);
    maskT = T == temperature;
    if i == 2
        maskT_a = maskT;
    elseif i == 3
        nexttile; hold on; box on;
        maskT = maskT | maskT_a;
        pd = fitdist(PTestGPR.Y{maskT,:}, 'kernel');
        plot(q, pdf(pd,q), '-k', 'LineWidth', lw);
        pd = fitdist(PTestGPR.YPred{maskT,1}, 'kernel');
        plot(q, pdf(pd,q), '-', 'Color', colortriplet(1,:), 'LineWidth', lw);
        pd = fitdist(PTestRF.YPred{maskT,1}, 'kernel');
        plot(q, pdf(pd,q), '-', 'Color', colortriplet(2,:), 'LineWidth', lw);
        pd = fitdist(PTestRF_2.YPred{maskT,1}, 'kernel');
        plot(q, pdf(pd,q), '-', 'Color', colortriplet(3,:), 'LineWidth', lw);
        xlabel('Rel. capacity');
        title('0 \circC, 10 \circC', 'FontWeight', 'normal');
    else
        nexttile; hold on; box on;
        pd = fitdist(PTestGPR.Y{maskT,:}, 'kernel');
        plot(q, pdf(pd,q), '-k', 'LineWidth', lw);
        pd = fitdist(PTestGPR.YPred{maskT,1}, 'kernel');
        plot(q, pdf(pd,q), '-', 'Color', colortriplet(1,:), 'LineWidth', lw);
        pd = fitdist(PTestRF.YPred{maskT,1}, 'kernel');
        plot(q, pdf(pd,q), '-', 'Color', colortriplet(2,:), 'LineWidth', lw);
        pd = fitdist(PTestRF_2.YPred{maskT,1}, 'kernel');
        plot(q, pdf(pd,q), '-', 'Color', colortriplet(3,:), 'LineWidth', lw);
        xlabel('Rel. capacity');
        title(sprintf('%d \\circC', temperature), 'FontWeight', 'normal');
    end
end
ylabel(t, 'Density');
set(gcf, 'Units', 'inches', 'Position', [8,2,6.5,2]);

annotation(gcf,'textbox',...
    [0.220551282051282 0.609375 0.0390641025641026 0.119791666666667],'String',{'a'},...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.449717948717947 0.614583333333333 0.0390641025641026 0.119791666666667],...
    'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.680487179487179 0.614583333333333 0.0390641025641025 0.119791666666667],...
    'String','c',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.91125641025641 0.614583333333333 0.0390641025641025 0.119791666666667],...
    'String','d',...
    'FitBoxToText','off',...
    'EdgeColor','none');

%% Helper methods
function idxKeep = filterInterpData(Data)
% If the value of Zmag at 100 Hz for the interpolated data is not within
% the range of Zmag at 10 Hz for all of the raw data for that
% temperature/cell, then get rid of that row.
idxKeep = true(height(Data), 1);
idxFreq = 36;
mask_m10C = Data.TdegC_EIS == -10;
mask_25C = Data.TdegC_EIS == 25;
uniqueSeries = unique(Data.seriesIdx, 'stable');
for thisSeries = uniqueSeries'
    maskSeries = Data.seriesIdx == thisSeries;
    % -10C
    maskSeries_m10C = maskSeries & mask_m10C;
    Zmag = Data.Zmag(maskSeries_m10C, idxFreq);
    idxKeep(maskSeries_m10C) = Zmag >= min(Zmag(~Data.isInterpEIS(maskSeries_m10C))) & Zmag <= max(Zmag(~Data.isInterpEIS(maskSeries_m10C)));
    % 25C
    maskSeries_25C = maskSeries & mask_25C;
    Zmag = Data.Zmag(maskSeries_25C, idxFreq);
    idxKeep(maskSeries_25C) = Zmag >= min(Zmag(~Data.isInterpEIS(maskSeries_25C))) & Zmag <= max(Zmag(~Data.isInterpEIS(maskSeries_25C)));
end
end

function Data = combineDataTables(DataFormatted, Data2Formatted)
% Need to add an isInterpEIS column to Data2Formatted to combine them.
Data2Formatted.isInterpEIS = zeros(height(Data2Formatted),1);
Data2Formatted = movevars(Data2Formatted, 'isInterpEIS', 'After', 'q');
% Make seriesIdx consistent for cells that have repeat data in DataFormatted 
% and Data2Formatted
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 32) = 1;
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 33) = 9;
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 34) = 21;
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 35) = 22;
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 36) = 26;
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 37) = 12;
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 38) = 13;
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 39) = 14;
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 40) = 18;
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 41) = 28;
Data2Formatted.seriesIdx(Data2Formatted.seriesIdx == 42) = 29;
% Combine the data tables
Data = [DataFormatted(:, [1, 17, 23:width(DataFormatted)]); Data2Formatted(:, [1,3:width(Data2Formatted)])];
Data = sortrows(Data, {'seriesIdx', 'TdegC_EIS', 'soc_EIS'});
end

function Pipes = fitPipes(Pipes, data, cvsplit, seriesIdxCV, seriesIdxTest)
wb = waitbar(0, "Fitting " + strrep(Pipes.name(1),"_"," "));
for i = 1:height(Pipes)
    waitbar(i/height(Pipes), wb, "Fitting " + strrep(Pipes.name(i),"_"," "));
    % Grab model
    M = Pipes.Model(i);
    % Train, cross-validate, and test
    [M, PTrain(i), Mcv, PCrossVal(i)] = crossvalidate(M, data.Xcv, data.Ycv, cvsplit, seriesIdxCV);
    PTest(i) = predict(M, data.Xtest, data.Ytest, seriesIdxTest);
    % Store results
    maeTrain(i) = PTrain(i).FitStats.mae;
    maeCrossVal(i) = PCrossVal(i).FitStats.mae;
    maeTest(i) = PTest(i).FitStats.mae;
end
close(wb)

Pipes.PTrain = PTrain';
Pipes.PCrossVal = PCrossVal';
Pipes.PTest = PTest';
Pipes.maeTrain = maeTrain';
Pipes.maeCrossVal = maeCrossVal';
Pipes.maeTest = maeTest';
end
