% Paul Gasper, NREL, 5/2022
clear; close all; clc
restoredefaultpath % Ensure this code runs entirely within this folder
addpath(genpath('functions'))

%{
This file runs searches for the best models by running a wide range of
model pipelines on several different splits of the data. Note that running
this file will take a long time (2 to 24 hours, depending on the selected
estimator). Detailed results can be saved, though the size of the data file 
can get quite large.

For a quicker example of using the RegressionPipeline to predict capacity
from impedance data, see the script 'example_Denso_models.m'. That script
trains several of the more interesting models, and visualizes some of the
results.
%}

% Filter out poorly interpolated data
%{
Match up seriesIdx for cells b/w the two data sets.
Split data set into 25C,50SOC and -10C,50SOC sets for training separate
models.
%}
% Load the data tables
load('data\Data_Denso2021.mat', 'Data', 'DataFormatted', 'Data2Formatted')
% Filter out noisy interpolated data
idxKeep = filterInterpData(Data);
%{
% Show that the filter is working
figure; plot(Data.Zreal(idxKeep,15:end), Data.Zimag(idxKeep,15:end), '.')
hold on; plot(Data.Zreal(~idxKeep,15:end), Data.Zimag(~idxKeep,15:end), 'k.')
% The filter removes a few measurements that might be okay, but all of the
% measurements that were clearly interpolated with error have been
% correctly removed from the data set.
%}
DataFormatted = DataFormatted(idxKeep, :);
DataAll = combineDataTables(DataFormatted, Data2Formatted);

% Load equivalent circuit model fit parameters as a separate table
DataECM = importfile("python/DensoData_EIS_paramters.csv");
DataECM = DataECM(idxKeep, :);
DataECM2 = importfile("python/DensoData_EIS_vSOCvT_paramters.csv");
% Add other variables to the ECM tables
DataECM = [DataFormatted(:,1:25), DataECM];
DataECM2 = [Data2Formatted(:,1:5), DataECM2];
% Combine ECM tables
DataAllECM = combineDataTables(DataECM, DataECM2);
% RC1 and RC2 order is not always right (RC1 should be at higher frequency than RC2):
DataAllECM = fixRCpairs(DataAllECM);

% Split into 25C and -10C, 50% SOC data sets for the initial modeling study
% -10C
mask_m10C = DataAll.TdegC_EIS == -10 & DataAll.soc_EIS == 0.5;
Data_m10C = DataAll(mask_m10C, :); DataECM_m10C = DataAllECM(mask_m10C, :);
% Remove BOL cell data (seriesIdx = 44, much different from aged cells)
Data_m10C = Data_m10C(1:end-1, :); DataECM_m10C = DataECM_m10C(1:end-1, :);
% 25C
mask_25C = DataAll.TdegC_EIS == 25 & DataAll.soc_EIS == 0.5;
Data_25C = DataAll(mask_25C, :); DataECM_25C = DataAllECM(mask_25C, :);
% Remove BOL cell data (seriesIdx = 43 & 44, much different from aged cells)
Data_25C = Data_25C(1:end-2, :); DataECM_25C = DataECM_25C(1:end-2, :);

% Clean up
clearvars Data DataFormatted Data2Formatted idxKeep DataECM DataECM2 mask_m10C mask_25C

%% -10C Data modeling
Data = Data_m10C; DataECM = DataECM_m10C;
% Pull out X and Y data tables. X variables are any Zreal, Zimag, Zmag, and
% Zphz data points. Y variable is the relative discharge capacity, q.
X = Data(:, 6:end); Y = Data(:,2); seriesIdx = Data{:, 1};
X_ECM = DataECM(:, 6:end); Y_ECM = DataECM(:,2);
% Data from cells running a WLTP drive cycle, and some from the aging study
% are used as test data.
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data.seriesIdx == cellsTest,2);
data.Xtest = X(maskTest,:); data.Ytest = Y(maskTest,:); 
dataECM.Xtest = X_ECM(maskTest,:); dataECM.Ytest = Y_ECM(maskTest,:);
seriesIdxTest = seriesIdx(maskTest);
% Cross-validation and training are conducted on the same data split.
data.Xcv = X(~maskTest,:); data.Ycv = Y(~maskTest,:); 
dataECM.Xcv = X_ECM(~maskTest,:); dataECM.Ycv = Y(~maskTest,:);
seriesIdxCV = seriesIdx(~maskTest);
% Define a cross-validation scheme.
cvsplit = cvpartseries(seriesIdxCV, 'Leaveout');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFINE ESTIMATOR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model training is done using RegressionPipeline objects. We will use one
% of several estimators: linear model, gaussian process model, and random
% forest model (in MATLAB, called a bagged tree ensemble).
estimator = @fitlm;
% estimator = @fitrgp;
% estimator = @fitrensemble;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the pipelines
TdegC = -10;
Pipes_m10C = defineModelPipelines(estimator, TdegC);

% Compare each of these models using the train/CV and test sets
warning('off')
Pipes_m10C = fitPipes(Pipes_m10C, data, dataECM, cvsplit, seriesIdxCV, seriesIdxTest);

%% 25C Data modeling
Data = Data_25C; DataECM = DataECM_25C;
% Pull out X and Y data tables. X variables are any Zreal, Zimag, Zmag, and
% Zphz data points. Y variable is the relative discharge capacity, q.
X = Data(:, 6:end); Y = Data(:,2); seriesIdx = Data{:, 1};
X_ECM = DataECM(:, 6:end); Y_ECM = DataECM(:,2);
% Data from cells running a WLTP drive cycle, and some from the aging study
% are used as test data.
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data.seriesIdx == cellsTest,2);
data.Xtest = X(maskTest,:); data.Ytest = Y(maskTest,:); 
dataECM.Xtest = X_ECM(maskTest,:); dataECM.Ytest = Y_ECM(maskTest,:);
seriesIdxTest = seriesIdx(maskTest);
% Cross-validation and training are conducted on the same data split.
data.Xcv = X(~maskTest,:); data.Ycv = Y(~maskTest,:); 
dataECM.Xcv = X_ECM(~maskTest,:); dataECM.Ycv = Y(~maskTest,:);
seriesIdxCV = seriesIdx(~maskTest);
% Define a cross-validation scheme.
cvsplit = cvpartseries(seriesIdxCV, 'Leaveout');

% Define the pipelines
TdegC = 25;
Pipes_25C = defineModelPipelines(estimator, TdegC);

% Compare each of these models using the train/CV and test sets
warning('off')
Pipes_25C = fitPipes(Pipes_25C, data, dataECM, cvsplit, seriesIdxCV, seriesIdxTest);

%% DCIR models
% Predict capacity using DC resistance metrics rather than AC impedance.
% Again, we have measurements at -10C and 25C.
load('data\Denso\DensoData.mat', 'DensoData');
DataDCIR = DensoData; clearvars DensoData;
% Clear out NaN rows
DataDCIR = DataDCIR(~isnan(DataDCIR.rm10C50soc10s), :);
% Features are resistance calculated using Ohm's law from pulse
% measurements at 50% SOC (there are also pulses at 20% and 70% SOC at
% 25C) from different time intervals after pulse start (0-0.01s, 0-0.1s,
% 0-10s). Use the average of the charge and discharge resistance for each
% time duration, since the charge/discharge resistances are very highly
% correlated anyways (0.95 to 0.99 correlation b/w charge and discharge).
dataVars = DataDCIR.Properties.VariableNames;
%25C 20soc
X25C20soc = DataDCIR(:, contains(dataVars, 'r25C20soc'));
%25C 50soc
X25C50soc = DataDCIR(:, contains(dataVars, 'r25C50soc'));
%25C 70soc
X25C70soc = DataDCIR(:, contains(dataVars, 'r25C70soc'));
%-10C 50soc
Xm10C50soc = DataDCIR(:, contains(dataVars, 'rm10C50soc'));
% capacity
Y = DataDCIR(:,'q');

% Same train/CV/test splits as above
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(DataDCIR.seriesIdx == cellsTest,2);
seriesIdxTest = DataDCIR.seriesIdx(maskTest);
seriesIdxCV = DataDCIR.seriesIdx(~maskTest);
cvsplit = cvpartseries(seriesIdxCV, 'Leaveout');
X25C20SOCcv = X25C20soc(~maskTest,:); X25C20SOCtest = X25C20soc(maskTest,:);  
X25C50SOCcv = X25C50soc(~maskTest,:); X25C50SOCtest = X25C50soc(maskTest,:);  
X25C70SOCcv = X25C70soc(~maskTest,:); X25C70SOCtest = X25C70soc(maskTest,:);  

Xm10Ccv = Xm10C50soc(~maskTest,:); Xm10Ctest = Xm10C50soc(maskTest,:);
Ycv = Y(~maskTest, :);  Ytest = Y(maskTest,:);

% Just use a straightforward GPR model.
seq = {@RegressionPipeline.normalizeZScore};
Model = RegressionPipeline(@fitrgp,...
    "FeatureTransformationSequence", seq);

disp("Fitting DCIR models")
% -10C
Pipes_DCIR(1).TdegC_DCIR = -10;
Pipes_DCIR(1).soc_DCIR = 0.5;
% Train, cross-validate, and test
[M, PredTrain, ~, PredCV] = crossvalidate(Model, Xm10Ccv, Ycv, cvsplit, seriesIdxCV);
PredTest = predict(M, Xm10Ctest, Ytest, seriesIdxTest);
Pipes_DCIR(1).Model = M;
Pipes_DCIR(1).PTrain = PredTrain;
Pipes_DCIR(1).PCrossVal = PredCV;
Pipes_DCIR(1).PTest = PredTest;
Pipes_DCIR(1).maeTrain = PredTrain.FitStats.mae;
Pipes_DCIR(1).maeCrossVal = PredCV.FitStats.mae;
Pipes_DCIR(1).maeTest = PredTest.FitStats.mae;

% 25C 20soc
Pipes_DCIR(2).TdegC_DCIR = 25;
Pipes_DCIR(2).soc_DCIR = 0.2;
% Train, cross-validate, and test
[M, PredTrain, ~, PredCV] = crossvalidate(Model, X25C20SOCcv, Ycv, cvsplit, seriesIdxCV);
PredTest = predict(M, X25C20SOCtest, Ytest, seriesIdxTest);
Pipes_DCIR(2).Model = M;
Pipes_DCIR(2).PTrain = PredTrain;
Pipes_DCIR(2).PCrossVal = PredCV;
Pipes_DCIR(2).PTest = PredTest;
Pipes_DCIR(2).maeTrain = PredTrain.FitStats.mae;
Pipes_DCIR(2).maeCrossVal = PredCV.FitStats.mae;
Pipes_DCIR(2).maeTest = PredTest.FitStats.mae;

% 25C 50soc
Pipes_DCIR(3).TdegC_DCIR = 25;
Pipes_DCIR(3).soc_DCIR = 0.5;
% Train, cross-validate, and test
[M, PredTrain, ~, PredCV] = crossvalidate(Model, X25C50SOCcv, Ycv, cvsplit, seriesIdxCV);
PredTest = predict(M, X25C50SOCtest, Ytest, seriesIdxTest);
Pipes_DCIR(3).Model = M;
Pipes_DCIR(3).PTrain = PredTrain;
Pipes_DCIR(3).PCrossVal = PredCV;
Pipes_DCIR(3).PTest = PredTest;
Pipes_DCIR(3).maeTrain = PredTrain.FitStats.mae;
Pipes_DCIR(3).maeCrossVal = PredCV.FitStats.mae;
Pipes_DCIR(3).maeTest = PredTest.FitStats.mae;

% 25C 50soc
Pipes_DCIR(4).TdegC_DCIR = 25;
Pipes_DCIR(4).soc_DCIR = 0.7;
% Train, cross-validate, and test
[M, PredTrain, ~, PredCV] = crossvalidate(Model, X25C70SOCcv, Ycv, cvsplit, seriesIdxCV);
PredTest = predict(M, X25C70SOCtest, Ytest, seriesIdxTest);
Pipes_DCIR(4).Model = M;
Pipes_DCIR(4).PTrain = PredTrain;
Pipes_DCIR(4).PCrossVal = PredCV;
Pipes_DCIR(4).PTest = PredTest;
Pipes_DCIR(4).maeTrain = PredTrain.FitStats.mae;
Pipes_DCIR(4).maeCrossVal = PredCV.FitStats.mae;
Pipes_DCIR(4).maeTest = PredTest.FitStats.mae;

%% Predict capacity using impedance from all temps, socs
% See if we can predict capacity accurately using impedance from a range of
% temperatures and SOCs.
% Load the data tables
load('data\Data_Denso2021.mat', 'Data', 'DataFormatted', 'Data2Formatted')
idxKeep = filterInterpData(Data);
DataFormatted = DataFormatted(idxKeep, :);
Data = combineDataTables(DataFormatted, Data2Formatted);
% Remove BOL data (seriesIdx = 43 & 44, the magnitude of the impedance is
% quite a bit different for these cells compared to the aging test matrix cells).
Data = Data(1:1194, :);
clearvars -except Data Pipes_m10C Pipes_25C Pipes_DCIR

% Grab X and Y data. Same train/cv/test splits as above.
X = Data(:, 6:end); Y = Data(:,2); seriesIdx = Data{:, 1};
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data.seriesIdx == cellsTest,2);
data.Xtest = X(maskTest,:);  data.Ytest = Y(maskTest,:);  seriesIdxTest = seriesIdx(maskTest);
data.Xcv   = X(~maskTest,:); data.Ycv   = Y(~maskTest,:); seriesIdxCV   = seriesIdx(~maskTest);
cvsplit = cvpartseries(seriesIdxCV, 'Leaveout');

% GPR and RandomForest models using 1) double frequency search, 2) SISSO, and 3) UMAP
%GPR
estimator = @fitrgp;
Pipes_GPR = defineModelPipelines2(estimator); warning('off')
Pipes_GPR = fitPipes(Pipes_GPR, data, [], cvsplit, seriesIdxCV, seriesIdxTest);
%RandomForest
estimator = @fitrensemble;
Pipes_RF = defineModelPipelines2(estimator); warning('off')
Pipes_RF = fitPipes(Pipes_RF, data, [], cvsplit, seriesIdxCV, seriesIdxTest);

%% Same as above, but use class weights to help handle imbalanced data
% Regression weights only work for linear or RF models. Use RF since it
% performs better than linear. GPR models do not allow weights, in MATLAB
% at least.

% Load the data tables
load('data\Data_Denso2021.mat', 'Data', 'DataFormatted', 'Data2Formatted')
idxKeep = filterInterpData(Data);
DataFormatted = DataFormatted(idxKeep, :);
Data = combineDataTables(DataFormatted, Data2Formatted);
% Remove BOL data (seriesIdx = 43 & 44, the magnitude of the impedance is
% quite a bit different for these cells compared to the aging test matrix cells).
Data = Data(1:1194, :);

% Grab X and Y data. Same train/cv/test splits as above.
X = Data(:, 6:end); Y = Data(:,2); seriesIdx = Data{:, 1};
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data.seriesIdx == cellsTest,2);
data.Xtest = X(maskTest,:);  data.Ytest = Y(maskTest,:);  seriesIdxTest = seriesIdx(maskTest);
data.Xcv   = X(~maskTest,:); data.Ycv   = Y(~maskTest,:); seriesIdxCV   = seriesIdx(~maskTest);
cvsplit = cvpartseries(seriesIdxCV, 'Leaveout');

% Calculate weights
weights = evenlyWeightDataSeries(Data.TdegC_EIS(~maskTest));

% RandomForest, double frequency search, weighted regression
estimator = @fitrensemble;
Pipes_RF_2 = defineModelPipelines3(estimator, weights); warning('off')
Pipes_RF_2 = fitPipes(Pipes_RF_2, data, [], cvsplit, seriesIdxCV, seriesIdxTest);

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

function Models = defineModelPipelines(estimator, TdegC)
% Model 0: Dummy regressor
Model_0 = RegressionPipeline(@fitrdummy);

% Model_1A: No feature extraction, no feature selection.
seq = {@RegressionPipeline.normalizeZScore};
Model_1A = RegressionPipeline(estimator,...
    "FeatureTransformationSequence", seq);
if strcmp(func2str(estimator), 'fitrensemble')
    Model_1A.ModelFuncOpts = {'Method','bag'};
end

% Models_1B: Only use a single frequency (4 features), no feature selection.
Models_1B = defineSingleFreqModels(estimator);

% Models_1C: Use two frequencies (8 features), no feature selection.
Models_1C = defineDoubleFreqModels(estimator);

% Models_1D: fscorr feature selection, 1:10 features, several cutoffs
Models_1D = defineFSCorrModels(estimator, 1:10, [0.99,0.98,0.95,0.9,0.8,0.65,0.5], "Model_1D");

% Models_1E: fssisso feature selection
% nNonzeroCoeffs = [2;3;4;5];
% nFeaturesPerSisIter = [100;30;7;3];
nNonzeroCoeffs = [2;3;4;5;6;7;8;9;10];
nFeaturesPerSisIter = [100;30;15;8;5;3;2;2;2];
Models_1E = defineFSSissoModels(estimator, nNonzeroCoeffs, nFeaturesPerSisIter, "Model_1E");

%Model_1F: embedded feature selection (lasso(?), ard kernel)
if strcmp(func2str(estimator), 'fitrgp')
    seq = {@RegressionPipeline.normalizeZScore};
    Model_1F = RegressionPipeline(estimator,...
        "ModelFuncOpts", {'KernelFunction','ardsquaredexponential'},...
        "FeatureTransformationSequence", seq);
elseif strcmp(func2str(estimator), 'fitlm')
    estimator2 = @fitrlinear;
    seq = {@RegressionPipeline.normalizeZScore};
    Model_1F = RegressionPipeline(estimator2,...
        "ModelFuncOpts", {'Regularization','lasso'},...
        "FeatureTransformationSequence", seq);
else
    Model_1F = [];
end

% Model_2A: Extract statistical features, no feature selection
seq = {@generateFeaturesStatistical,...
    @RegressionPipeline.normalizeZScore};
Model_2A = RegressionPipeline(estimator,...
    "FeatureTransformationSequence", seq);
if strcmp(func2str(estimator), 'fitrensemble')
    Model_2A.ModelFuncOpts = {'Method','bag'};
end
% Model 2B: Extract statistical features, use fscorr (1:10 features, 95% cutoff)
Models_2B = defineFSCorrModels2(estimator, 10, 0.95, "Model_2B");
% Model 2C: Extract statistical features, use fssisso
nNonzeroCoeffs = [2;3;4;5];
nFeaturesPerSisIter = [14;9;7;3];
Models_2C = defineFSSissoModels2(estimator, nNonzeroCoeffs, nFeaturesPerSisIter, "Model_2C");

% Model_3a: Extract PCA features, no feature selection
% Note - PCA is sensitive to the magnitude of each column. Normalize first.
% Use only Zreal and Zimag, since Zmag and Zphz are derived from them.
% Manual exploration shows that 10 features explains 99.3% of the variance
% in X.
seq = {@RegressionPipeline.normalizeZScore,...
    @selectOnlyZrealZimag,...
    @RegressionPipeline.generateFeaturesPCA};
hyp = {{},{},...
    {"n",10,"KeepPriorFeatures",false}};
Model_3A = RegressionPipeline(estimator,...
    "FeatureTransformationSequence", seq,...
    "FeatureTransformationFixedHyp", hyp);
if strcmp(func2str(estimator), 'fitrensemble')
    Model_3A.ModelFuncOpts = {'Method','bag'};
end

% Models_3b: Extract UMAP features, no feature selection
n = [2;3;4;5];
nNeighbors = [15;20;30;50;100];
minDist = [0.01;0.03;0.06;0.1;0.3];
Models_3B = defineUmapModels(estimator, n, nNeighbors, minDist, "Models_3B");

% Model 4: Extract graphical features, no feature selection
if TdegC == -10
    seq = {@generateFeaturesGraphicalM10C,...
        @RegressionPipeline.normalizeZScore};
elseif TdegC == 25
    seq = {@generateFeaturesGraphical25C,...
        @RegressionPipeline.normalizeZScore};
else
    error('Help!')
end
Model_4 = RegressionPipeline(estimator,...
    "FeatureTransformationSequence", seq);
if strcmp(func2str(estimator), 'fitrensemble')
    Model_4.ModelFuncOpts = {'Method','bag'};
end

% Model 5: ECM features, no feature selection
seq = {@RegressionPipeline.normalizeZScore};
Model_5 = RegressionPipeline(estimator,...
    "FeatureTransformationSequence", seq);
if strcmp(func2str(estimator), 'fitrensemble')
    Model_5.ModelFuncOpts = {'Method','bag'};
end

% Return model objects
name = ["Model_0";...
    "Model_1A"; "Models_1B"; "Models_1C"; "Models_1D"; "Models_1E"; "Model_1F";...
    "Model_2A"; "Models_2B"; "Models_2C";...
    "Model_3A"; "Models_3B"; "Model_4"; "Model_5"];
Model = {Model_0;...
    Model_1A; Models_1B; Models_1C; Models_1D; Models_1E; Model_1F;...
    Model_2A; Models_2B; Models_2C;...
    Model_3A; Models_3B; Model_4; Model_5};
Models = table(name, Model);
end

function Models = defineSingleFreqModels(estimator)
name = strings(69, 1);
idxFreq = [1:69]';
for i = 1:69
    seq = {@RegressionPipeline.normalizeZScore,...
        @selectFrequency};
    hyp = {{}, {"idxFreq", idxFreq(i)}};
    M = RegressionPipeline(estimator,...
        "FeatureTransformationSequence", seq,...
        "FeatureTransformationFixedHyp", hyp);
    name(i) = "Model_1B_" + i;
    if strcmp(func2str(estimator), 'fitrensemble')
        M.ModelFuncOpts = {'Method','bag'};
    end
    Model(i) = M;
end
Model = transpose(Model);
Models = table(name, idxFreq, Model);
end

function Models = defineDoubleFreqModels(estimator, weights)
idxFreq = nchoosek([1:69],2);
name = strings(size(idxFreq, 1), 1);
for i = 1:size(idxFreq, 1)
    seq = {@RegressionPipeline.normalizeZScore,...
        @selectFrequency};
    hyp = {{}, {"idxFreq", idxFreq(i,:)}};
    if nargin == 2
        ModelOpts.Weights = weights;
        M = RegressionPipeline(estimator,...
            "FeatureTransformationSequence", seq,...
            "FeatureTransformationFixedHyp", hyp,...
            "ModelOpts", ModelOpts);
    else
        M = RegressionPipeline(estimator,...
            "FeatureTransformationSequence", seq,...
            "FeatureTransformationFixedHyp", hyp);
    end
    name(i) = "Model_1C_" + i;
    if strcmp(func2str(estimator), 'fitrensemble')
        M.ModelFuncOpts = {'Method','bag'};
    end
    Model(i) = M;
end
Model = transpose(Model);
Models = table(name, idxFreq, Model);
end

function Models = defineFSCorrModels(estimator, n, p, namePrefix)
name = strings(length(n)*length(p), 1);
idx = 1;
for i = 1:length(n)
    for ii = 1:length(p)
        seq = {@RegressionPipeline.normalizeZScore,...
            @fscorr};
        hyp = {{}, {"y", [], "n", n(i), "p", p(ii)}};
        M = RegressionPipeline(estimator,...
            "FeatureTransformationSequence", seq,...
            "FeatureTransformationFixedHyp", hyp);
        name(idx) = namePrefix + "_" + idx;
        Model(idx) = M;
        if strcmp(func2str(estimator), 'fitrensemble')
            M.ModelFuncOpts = {'Method','bag'};
        end
        idx = idx+1;
    end
end
if isrow(p); p = p'; end
p_ = repmat(p, length(n), 1);
n = repmat(n, length(p), 1);
n = reshape(n, [], 1);
p = p_;
Model = transpose(Model);
Models = table(name, n, p, Model);
end

function Models = defineFSCorrModels2(estimator, n, p, namePrefix)
name = strings(n, 1);
for i = 1:n
    seq = {@generateFeaturesStatistical,...
        @RegressionPipeline.normalizeZScore,...
        @fscorr};
    hyp = {{}, {}, {"y", [], "n", i, "p", p}};
    M = RegressionPipeline(estimator,...
        "FeatureTransformationSequence", seq,...
        "FeatureTransformationFixedHyp", hyp);
    name(i) = namePrefix + "_" + i;
    if strcmp(func2str(estimator), 'fitrensemble')
        M.ModelFuncOpts = {'Method','bag'};
    end
    Model(i) = M;
end
p = repmat(p, n, 1);
n = [1:n]';
Model = transpose(Model);
Models = table(name, n, p, Model);
end

function Models = defineFSSissoModels(estimator, nNonzeroCoeffs, nFeaturesPerSisIter, namePrefix)
name = strings(length(nNonzeroCoeffs), 1);
for i = 1:length(nNonzeroCoeffs)
    seq = {@RegressionPipeline.normalizeZScore,...
        @fssisso};
    hyp = {{}, {"y", [],...
        "nNonzeroCoeffs", nNonzeroCoeffs(i),...
        "nFeaturesPerSisIter", nFeaturesPerSisIter(i)}};
    M = RegressionPipeline(estimator,...
        "FeatureTransformationSequence", seq,...
        "FeatureTransformationFixedHyp", hyp);
    name(i) = namePrefix + "_" + i;
    if strcmp(func2str(estimator), 'fitrensemble')
        M.ModelFuncOpts = {'Method','bag'};
    end
    Model(i) = M;
end
Model = transpose(Model);
Models = table(name, nNonzeroCoeffs, nFeaturesPerSisIter, Model);
end

function Models = defineFSSissoModels2(estimator, nNonzeroCoeffs, nFeaturesPerSisIter, namePrefix)
name = strings(length(nNonzeroCoeffs), 1);
for i = 1:length(nNonzeroCoeffs)
    seq = {@generateFeaturesStatistical,...
        @RegressionPipeline.normalizeZScore,...
        @fssisso};
    hyp = {{}, {}, {"y", [],...
        "nNonzeroCoeffs", nNonzeroCoeffs(i),...
        "nFeaturesPerSisIter", nFeaturesPerSisIter(i)}};
    M = RegressionPipeline(estimator,...
        "FeatureTransformationSequence", seq,...
        "FeatureTransformationFixedHyp", hyp);
    name(i) = namePrefix + "_" + i;
    if strcmp(func2str(estimator), 'fitrensemble')
        M.ModelFuncOpts = {'Method','bag'};
    end
    Model(i) = M;
end
Model = transpose(Model);
Models = table(name, nNonzeroCoeffs, nFeaturesPerSisIter, Model);
end

function Models = defineUmapModels(estimator, n, nNeighbors, minDist, namePrefix)
len = length(n)*length(nNeighbors)*length(minDist);
name = strings(len, 1);
n_ = zeros(len, 1);
nNeighbors_ = zeros(len, 1);
minDist_ = zeros(len, 1);
idx = 1;
for i = 1:length(n)
    for ii = 1:length(nNeighbors)
        for iii = 1:length(minDist)
            seq = {@RegressionPipeline.normalizeZScore,...
                @selectOnlyZrealZimag,...
                @generateFeaturesUMAP};
            hyp = {{},{},...
                {"n",n(i),"nNeighbors",nNeighbors(ii),"minDist",minDist(iii),"KeepPriorFeatures",false}};
            M = RegressionPipeline(estimator,...
                "FeatureTransformationSequence", seq,...
                "FeatureTransformationFixedHyp", hyp);
            if strcmp(func2str(estimator), 'fitrensemble')
                M.ModelFuncOpts = {'Method','bag'};
            end
            n_(idx) = n(i);
            nNeighbors_(idx) = nNeighbors(ii);
            minDist_(idx) = minDist(iii);
            Model(idx) = M;
            name(idx) = namePrefix + "_" + idx;
            idx = idx+1;
        end
    end
end
n = n_; nNeighbors = nNeighbors_; minDist = minDist_;
Model = transpose(Model);
Models = table(name, n, nNeighbors, minDist, Model);
end

function Pipes = fitPipes(Pipes, data, dataECM, cvsplit, seriesIdxCV, seriesIdxTest)
PTrain = cell(height(Pipes), 1);
PCrossVal = cell(height(Pipes), 1);
PTest = cell(height(Pipes), 1);
maeTrain = cell(height(Pipes), 1);
maeCrossVal = cell(height(Pipes), 1);
maeTest = cell(height(Pipes), 1);
idxMinTrain = zeros(height(Pipes),1);
idxMinCV = zeros(height(Pipes),1);
idxMinTest = zeros(height(Pipes),1);
for i = 1:height(Pipes)
    disp("Fitting " + Pipes.name(i))
    M = Pipes.Model{i};
    if isa(M, 'RegressionPipeline')
        % Train, cross-validate, and test
        if ~strcmp(Pipes.name(i), "Model_5")
            [M, PredTrain, ~, PredCV] = crossvalidate(M, data.Xcv, data.Ycv, cvsplit, seriesIdxCV);
            PredTest = predict(M, data.Xtest, data.Ytest, seriesIdxTest);
        else
            [M, PredTrain, ~, PredCV] = crossvalidate(M, dataECM.Xcv, dataECM.Ycv, cvsplit, seriesIdxCV);
            PredTest = predict(M, dataECM.Xtest, dataECM.Ytest, seriesIdxTest);
        end
        % Store results
        Pipes.Model{i} = M;
        PTrain{i} = PredTrain;
        PCrossVal{i} = PredCV;
        PTest{i} = PredTest;
        maeTrain{i} = PredTrain.FitStats.mae;
        maeCrossVal{i} = PredCV.FitStats.mae;
        maeTest{i} = PredTest.FitStats.mae;
        idxMinTrain(i) = 1;
        idxMinCV(i) = 1;
        idxMinTest(i) = 1;
        clearvars PredTrain PredCV PredTest
    elseif istable(M) % table of pipelines
        Pipes2 = M;
        errTrain = zeros(height(Pipes2),1);
        errCV = zeros(height(Pipes2),1);
        errTest = zeros(height(Pipes2),1);
        flagFeatureSelection = any(strcmp(Pipes.name(i), ["Models_1D","Models_1E","Models_2B","Models_2C"]));
        if flagFeatureSelection
            idxSelected = cell(height(Pipes2),1);
        end
        flagKeepModels = any(strcmp(Pipes.name(i), ["Models_1D","Models_1E"]));
        if flagKeepModels
            ModelsCV = cell(height(Pipes2),1);
        end
        wb = waitbar(0, "Fitting " + strrep(Pipes.name(i),"_"," ") + ", 0 of " + height(Pipes2));
        for ii = 1:height(Pipes2)
            waitbar(ii/height(Pipes2), wb, "Fitting " + strrep(Pipes.name(i),"_"," ") + ", " + ii + " of " + height(Pipes2));
            % Grab model
            M = Pipes2.Model(ii);
            % Train, cross-validate, and test
            [M, PredTrain(ii), Mcv, PredCV(ii)] = crossvalidate(M, data.Xcv, data.Ycv, cvsplit, seriesIdxCV);
            PredTest(ii) = predict(M, data.Xtest, data.Ytest, seriesIdxTest);
            % Store results
            errTrain(ii) = PredTrain(ii).FitStats.mae;
            errCV(ii) = PredCV(ii).FitStats.mae;
            errTest(ii) = PredTest(ii).FitStats.mae;
            if flagFeatureSelection
                if contains(Pipes.name(i), "Models_1")
                    hyp = M.FeatureTransformationTrainedHyp{2};
                    idxSelected{ii} = hyp{2};
                elseif contains(Pipes.name(i), "Models_2")
                    hyp = M.FeatureTransformationTrainedHyp{3};
                    idxSelected{ii} = hyp{2};
                end
            end
            if flagKeepModels
                ModelsCV{ii} = Mcv;
                Pipes2.Model(ii) = M;
            end
        end
        close(wb)
        if flagFeatureSelection
            Pipes2.idxSelected = idxSelected;
        end
        if flagKeepModels
            Pipes2.ModelsCV = ModelsCV;
        else
            Pipes2 = removevars(Pipes2, 'Model');
        end
        [~, idxMinTrain(i)] = min(errTrain);
        [~, idxMinCV(i)] = min(errCV);
        [~, idxMinTest(i)] = min(errTest);
        Pipes.Model{i} = Pipes2;
        PTrain{i} = PredTrain([idxMinTrain(i), idxMinCV(i), idxMinTest(i)]);
        PCrossVal{i} = PredCV([idxMinTrain(i), idxMinCV(i), idxMinTest(i)]);
        PTest{i} = PredTest([idxMinTrain(i), idxMinCV(i), idxMinTest(i)]);
        maeTrain{i} = errTrain;
        maeCrossVal{i} = errCV;
        maeTest{i} = errTest;
        clearvars PredTrain PredCV PredTest
    else
        % empty (Model_1F, regression tree)
        Pipes.Model{i} = [];
        PTrain{i} = [];
        PCrossVal{i} = [];
        PTest{i} = [];
        maeTrain{i} = [];
        maeCrossVal{i} = [];
        maeTest{i} = [];
    end
end
Pipes.PTrain = PTrain;
Pipes.PCrossVal = PCrossVal;
Pipes.PTest = PTest;
Pipes.maeTrain = maeTrain;
Pipes.maeCrossVal = maeCrossVal;
Pipes.maeTest = maeTest;
Pipes.idxMinTrain = idxMinTrain;
Pipes.idxMinCV = idxMinCV;
Pipes.idxMinTest = idxMinTest;
end

function Pipes = fitPipes2(Pipes, data, seriesIdxTrain, seriesIdxTest)
PTrain = cell(height(Pipes), 1);
PTest = cell(height(Pipes), 1);
maeTrain = cell(height(Pipes), 1);
maeTest = cell(height(Pipes), 1);
idxMinTrain = zeros(height(Pipes),1);
idxMinTest = zeros(height(Pipes),1);
for i = 1:height(Pipes)
    disp("Fitting " + Pipes.name(i))
    M = Pipes.Model{i};
    if isa(M, 'RegressionPipeline')
        % Train, cross-validate, and test
        [M, PredTrain] = train(M, data.Xtrain, data.Ytrain, seriesIdxTrain);
        PredTest = predict(M, data.Xtest, data.Ytest, seriesIdxTest);

        % Store results
        Pipes.Model{i} = M;
        PTrain{i} = PredTrain;
        PTest{i} = PredTest;
        maeTrain{i} = PredTrain.FitStats.mae;
        maeTest{i} = PredTest.FitStats.mae;
        idxMinTrain(i) = 1;
        idxMinTest(i) = 1;
        clearvars PredTrain PredCV PredTest
    elseif istable(M) % table of pipelines
        Pipes2 = M;
        errTrain = zeros(height(Pipes2),1);
        errTest = zeros(height(Pipes2),1);
        flagFeatureSelection = any(strcmp(Pipes.name(i), ["Models_1D","Models_1E","Models_2B","Models_2C"]));
        if flagFeatureSelection
            idxSelected = cell(height(Pipes2),1);
        end
        flagKeepModels = any(strcmp(Pipes.name(i), ["Models_1D","Models_1E"]));
        if flagKeepModels
            ModelsCV = cell(height(Pipes2),1);
        end
        wb = waitbar(0, "Fitting " + strrep(Pipes.name(i),"_"," ") + ", 0 of " + height(Pipes2));
        for ii = 1:height(Pipes2)
            waitbar(ii/height(Pipes2), wb, "Fitting " + strrep(Pipes.name(i),"_"," ") + ", " + ii + " of " + height(Pipes2));
            % Grab model
            M = Pipes2.Model(ii);
            % Train and test
            [M, PredTrain(ii)] = train(M, data.Xtrain, data.Ytrain, seriesIdxTrain);
            PredTest(ii) = predict(M, data.Xtest, data.Ytest, seriesIdxTest);
            % Store results
            errTrain(ii) = PredTrain(ii).FitStats.mae;
            errTest(ii) = PredTest(ii).FitStats.mae;
            if flagFeatureSelection
                if contains(Pipes.name(i), "Models_1")
                    hyp = M.FeatureTransformationTrainedHyp{2};
                    idxSelected{ii} = hyp{2};
                elseif contains(Pipes.name(i), "Models_2")
                    hyp = M.FeatureTransformationTrainedHyp{3};
                    idxSelected{ii} = hyp{2};
                end
            end
            if flagKeepModels
                Pipes2.Model(ii) = M;
            end
        end
        close(wb)
        if flagFeatureSelection
            Pipes2.idxSelected = idxSelected;
        end
        [~, idxMinTrain(i)] = min(errTrain);
        [~, idxMinTest(i)] = min(errTest);
        Pipes.Model{i} = Pipes2;
        PTrain{i} = PredTrain([idxMinTrain(i), idxMinTest(i)]);
        PTest{i} = PredTest([idxMinTrain(i), idxMinTest(i)]);
        maeTrain{i} = errTrain;
        maeTest{i} = errTest;
        clearvars PredTrain PredCV PredTest
    else
        % empty (Model_1F, regression tree)
        Pipes.Model{i} = [];
        PTrain{i} = [];
        PTest{i} = [];
        maeTrain{i} = [];
        maeTest{i} = [];
    end
end
Pipes.PTrain = PTrain;
Pipes.PTest = PTest;
Pipes.maeTrain = maeTrain;
Pipes.maeTest = maeTest;
Pipes.idxMinTrain = idxMinTrain;
Pipes.idxMinTest = idxMinTest;
end

function DataECM = importfile(filename)
%IMPORTFILE Import data from a text file
%  DENSODATAEISPARAMTERS = IMPORTFILE(FILENAME) reads data from text
%  file FILENAME for the default selection.  Returns the data as a table.
%
%  DENSODATAEISPARAMTERS = IMPORTFILE(FILE, DATALINES) reads data for
%  the specified row interval(s) of text file FILENAME. Specify
%  DATALINES as a positive scalar integer or a N-by-2 array of positive
%  scalar integers for dis-contiguous row intervals.
%
%  Example:
%  DensoDataEISparamters = importfile("C:\Users\pgasper\Documents\GitHub\denso_eis_pipeline\python\DensoData_EIS_paramters.csv", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 18-Mar-2022 10:27:10

%% Input handling
dataLines = [2, Inf];

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 9);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["L0", "R0", "R1", "C1", "R2", "C2", "Wo2_0", "Wo2_1", "C3"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
DataECM = readtable(filename, opts);
end

function DataECM = fixRCpairs(DataECM)
tau1 = DataECM.R1.*DataECM.C1;
tau2 = DataECM.R2.*DataECM.C2;
maskFlip = tau1 > tau2;
r1c1 = zeros(height(DataECM), 2);
r2c2 = zeros(height(DataECM), 2);
r1c1(~maskFlip, :) = DataECM{~maskFlip, {'R1', 'C1'}};
r1c1(maskFlip, :) =  DataECM{maskFlip,  {'R2', 'C2'}};
r2c2(~maskFlip, :) = DataECM{~maskFlip, {'R2', 'C2'}};
r2c2(maskFlip, :) =  DataECM{maskFlip,  {'R1', 'C1'}};
DataECM.R1 = r1c1(:,1);
DataECM.C1 = r1c1(:,2);
DataECM.R2 = r2c2(:,1);
DataECM.C2 = r2c2(:,2);
end

function Models = defineModelPipelines2(estimator)
% Model 0: Dummy regressor
Model_0 = RegressionPipeline(@fitrdummy);

% Models_1A: Use two frequencies (8 features), no feature selection.
Models_1A = defineDoubleFreqModels(estimator);

% Models_1B: fssisso feature selection
nNonzeroCoeffs = [2;3;4;5;6;7;8;9;10];
nFeaturesPerSisIter = [100;30;15;8;5;3;2;2;2];
Models_1B = defineFSSissoModels(estimator, nNonzeroCoeffs, nFeaturesPerSisIter, "Model_1B");

% Models_2: Extract UMAP features, no feature selection
n = [2;3;4;5];
nNeighbors = [15;20;30;50];
minDist = [0.03;0.06;0.1;0.3];
Models_2 = defineUmapModels(estimator, n, nNeighbors, minDist, "Models_2");

% Return model objects
name = ["Model_0"; "Models_1A"; "Models_1B"; "Models_2"];
Model = {Model_0;   Models_1A;   Models_1B;   Models_2};
Models = table(name, Model);

% % Return model objects
% name = ["Model_0"; "Models_1A";];
% Model = {Model_0;   Models_1A;};
% Models = table(name, Model);
end

function Models = defineModelPipelines3(estimator, weights)
% Model 0: Dummy regressor
Model_0 = RegressionPipeline(@fitrdummy);

% Models_1A: Use two frequencies (8 features), no feature selection.
Models_1A = defineDoubleFreqModels(estimator, weights);

% Return model objects
name = ["Model_0"; "Models_1A";];
Model = {Model_0;   Models_1A;};
Models = table(name, Model);
end

