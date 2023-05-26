clear; clc; close all;
addpath(genpath('.'))

load('data\Data_FarajiNiri.mat', 'Data')

%% Train and cross-validate models
% Split off features and target variables
X = Data(:, 5:end); Y = Data(:,2); seriesIdx = Data{:, 1};
% Use 5-fold cross-validation like the Faraji-Niri paper
cvsplit = cvpartseries(seriesIdx, 'KFold', 5);

% Define Linear models: baseline, single frequency, two frequencies
Pipes_Linear = defineModelPipelines(@fitlm);
% Run cross-validation across all models
warning('off')
Pipes_Linear = fitPipes(Pipes_Linear, X, Y, cvsplit, seriesIdx);

% Define GPR models: baseline, single frequency, two frequencies
Pipes_GPR = defineModelPipelines(@fitrgp);
% Run cross-validation across all models
warning('off')
Pipes_GPR = fitPipes(Pipes_GPR, X, Y, cvsplit, seriesIdx);

%% MAE vs. frequency for single freq models
% Load frequency vector
load('data\Faraji-Niri_2023\WholeDataRealSOH.mat', 'WholeDataRealSOH')
eis = WholeDataRealSOH.EIS{1};
freq = eis(:,1);

% cv and test error versus model type
maeCrossValLinear = Pipes_Linear.maeCrossVal{2};
maeCrossValGpr = Pipes_GPR.maeCrossVal{2};

% plot
figure; t=tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; grid on; box on; hold on;
yline(Pipes_Linear.maeCrossVal{1}, '--k', 'LineWidth', 1.5)
plot(freq, maeCrossValLinear, '-', 'LineWidth', 1.5)
plot(freq, maeCrossValGpr, '-', 'LineWidth', 1.5)
% decorations
legend("Baseline", "Linear", "GPR")
set(gca, 'XScale', 'log'); ylim([0 0.08])
%xticks([1e-2 1e0 1e2 1e4]); 
set(gca, 'XMinorGrid', 'Off')
xlabel('Frequency (Hz)');
ylabel('MAE (5-fold CV)')

%% MAE vs frequency for double freq models
% 2D contour plot of MAE test vs. frequency selection
% Get two frequency model frequency indices array
idxFreq = nchoosek([1:61],2);

% Get MAE vs frequency indices
maeCVLinear = Pipes_Linear.maeCrossVal{3};
maeCVGpr = Pipes_GPR.maeCrossVal{3};
% colors
cmap = viridis(256); %cmap = [cmap; repmat(cmap(end,:), 300, 1)];
% levels for contours
levels = 0.02:0.002:0.07;

% Linear
z = nan(length(freq), length(freq));
for i = 1:size(idxFreq,1)
    z(idxFreq(i,1),idxFreq(i,2)) = maeCVLinear(i);
end
[x,y] = meshgrid(freq,freq);
figure; tiledlayout('flow'); nexttile;
contourf(y,x,z,levels); colormap(cmap);
cb = colorbar(); cb.Label.String = "MAE_{CrossVal}"; cb.Layout.Tile = 'east';
cb.Label.Position = [-1.183333396911621,0.038602659295312,0];
set(gca, 'XScale', 'log', 'Yscale', 'log')
hold on; box on; grid on;
[~, idxBest] = min(maeCVLinear);
idxBest = idxFreq(idxBest,:);
fx = freq(idxBest(1)); fy = freq(idxBest(2));
plot(fx, fy, 'ok', 'MarkerFaceColor', 'r', 'MarkerSize', 10)
plot([fx,fx],[min(freq),max(freq)], '-r')
plot([min(freq),max(freq)],[fy,fy], '-r')
xlabel('Frequency 1 (Hz)'); ylabel('Frequency 2 (Hz)');
xticks([1e-2 1e0 1e2 1e4]); yticks([1e-2 1e0 1e2 1e4]); 
set(gcf, 'Units', 'inches', 'Position', [3,3,3.25,2.5])
annotation(gcf,'textbox',...
    [0.202991452991453 0.686111111111111 0.273504273504274 0.158333333333333],...
    'String',sprintf("Minimum MAE: %0.2g", min(maeCVLinear)),...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName', 'Arial');
set(gca, 'FontName', 'Arial')

% Gpr
z = nan(length(freq), length(freq));
for i = 1:size(idxFreq,1)
    z(idxFreq(i,1),idxFreq(i,2)) = maeCVGpr(i);
end
[x,y] = meshgrid(freq,freq);
figure; tiledlayout('flow'); nexttile;
contourf(y,x,z,levels); colormap(cmap);
cb = colorbar(); cb.Label.String = "MAE_{CrossVal}"; cb.Layout.Tile = 'east';
cb.Label.Position = [-1.183333396911621,0.035602659295312,0];
set(gca, 'XScale', 'log', 'Yscale', 'log')
hold on; box on; grid on;
[~, idxBest] = min(maeCVGpr);
idxBest = idxFreq(idxBest,:);
fx = freq(idxBest(1)); fy = freq(idxBest(2));
plot(fx, fy, 'ok', 'MarkerFaceColor', 'r', 'MarkerSize', 10)
plot([fx,fx],[min(freq),max(freq)], '-r')
plot([min(freq),max(freq)],[fy,fy], '-r')
xlabel('Frequency 1 (Hz)'); ylabel('Frequency 2 (Hz)');
xticks([1e-2 1e0 1e2 1e4]); yticks([1e-2 1e0 1e2 1e4]); 
set(gcf, 'Units', 'inches', 'Position', [3,3,3.25,2.5])
annotation(gcf,'textbox',...
    [0.202991452991453 0.686111111111111 0.273504273504274 0.158333333333333],...
    'String',sprintf("Minimum MAE: %0.2g", min(maeCVGpr)),...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName', 'Arial');
set(gca, 'FontName', 'Arial')

%% Parity plot of best model
Pred = Pipes_GPR.PCrossVal{3}; Pred = Pred(2);
figure; hold on; box on; grid on;
plot(Pred.Y{:,:}, Pred.YPred{:,1}, 'ok')
axis square;
axis([0.75 1.05 0.75 1.05])
plot([0.75 1.05], [0.75 1.05], '-k')
xlabel('Rel. discharge capacity')
ylabel('Predicted rel. discharge capacity (5-fold CV)')


%% Helper methods
function Models = defineModelPipelines(estimator)
% Model 0: Dummy regressor
Model_0 = RegressionPipeline(@fitrdummy);
% Models_1B: Only use a single frequency (4 features), no feature selection.
Models_1B = defineSingleFreqModels(estimator);
% Models_1C: Use two frequencies (8 features), no feature selection.
Models_1C = defineDoubleFreqModels(estimator);
name = ["Model_0"; "Models_1B"; "Models_1C"];
Model = {Model_0; Models_1B; Models_1C};
Models = table(name, Model);
end

function Models = defineSingleFreqModels(estimator)
name = strings(61, 1);
idxFreq = [1:61]';
for i = 1:61
    seq = {@RegressionPipeline.normalizeZScore,...
        @selectFrequency_FarajiNiri};
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
idxFreq = nchoosek([1:61],2);
name = strings(size(idxFreq, 1), 1);
for i = 1:size(idxFreq, 1)
    seq = {@RegressionPipeline.normalizeZScore,...
        @selectFrequency_FarajiNiri};
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

function Pipes = fitPipes(Pipes, X, Y, cvsplit, seriesIdx)
PTrain = cell(height(Pipes), 1);
PCrossVal = cell(height(Pipes), 1);
maeTrain = cell(height(Pipes), 1);
maeCrossVal = cell(height(Pipes), 1);
idxMinTrain = zeros(height(Pipes),1);
idxMinCV = zeros(height(Pipes),1);
for i = 1:height(Pipes)
    disp("Fitting " + Pipes.name(i))
    M = Pipes.Model{i};
    if isa(M, 'RegressionPipeline')
        % Train, cross-validate, and test
        [M, PredTrain, ~, PredCV] = crossvalidate(M, X, Y, cvsplit, seriesIdx);
        % Store results
        Pipes.Model{i} = M;
        PTrain{i} = PredTrain;
        PCrossVal{i} = PredCV;
        maeTrain{i} = PredTrain.FitStats.mae;
        maeCrossVal{i} = PredCV.FitStats.mae;
        idxMinTrain(i) = 1;
        idxMinCV(i) = 1;
        clearvars PredTrain PredCV PredTest
    elseif istable(M) % table of pipelines
        Pipes2 = M;
        errTrain = zeros(height(Pipes2),1);
        errCV = zeros(height(Pipes2),1);
        wb = waitbar(0, "Fitting " + strrep(Pipes.name(i),"_"," ") + ", 0 of " + height(Pipes2));
        for ii = 1:height(Pipes2)
            waitbar(ii/height(Pipes2), wb, "Fitting " + strrep(Pipes.name(i),"_"," ") + ", " + ii + " of " + height(Pipes2));
            % Grab model
            M = Pipes2.Model(ii);
            % Train, cross-validate, and test
            [M, PredTrain(ii), Mcv, PredCV(ii)] = crossvalidate(M, X, Y, cvsplit, seriesIdx);
            % Store results
            errTrain(ii) = PredTrain(ii).FitStats.mae;
            errCV(ii) = PredCV(ii).FitStats.mae;
        end
        close(wb)
        [~, idxMinTrain(i)] = min(errTrain);
        [~, idxMinCV(i)] = min(errCV);
        Pipes.Model{i} = Pipes2;
        PTrain{i} = PredTrain([idxMinTrain(i), idxMinCV(i)]);
        PCrossVal{i} = PredCV([idxMinTrain(i), idxMinCV(i)]);
        maeTrain{i} = errTrain;
        maeCrossVal{i} = errCV;
        clearvars PredTrain PredCV PredTest
    end
end
Pipes.PTrain = PTrain;
Pipes.PCrossVal = PCrossVal;
Pipes.maeTrain = maeTrain;
Pipes.maeCrossVal = maeCrossVal;
Pipes.idxMinTrain = idxMinTrain;
Pipes.idxMinCV = idxMinCV;
end