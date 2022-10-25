%% Results analysis
addpath('C:\Users\pgasper\Documents\GitHub\ml_utils_matlab')
addpath('C:\Users\pgasper\Documents\GitHub\SISSORegressor_MATLAB')
addpath(genpath('C:\Users\pgasper\Documents\GitHub\colormaps_matlab'))
addpath('functions')

% freq
load('data\Data_Denso2021.mat', 'Data')
freq = Data.Freq(1,:); clearvars Data

% set of three colors for plots
colors1 = brewermap('RdPu', 22);
colors1 = colors1(end-10:end,:);
colors2 = brewermap('Blues', 38);
colors2 = colors2(end-17:end,:);
colors3 = brewermap('Greens', 5);
colors3 = colors3(end-1:end,:);
colortriplet = [colors1(3,:); colors2(12,:); colors3(2,:)];

% line colors
cellsTest = [7,10,13,17,24,30,31];
colors = [colors1;colors2;colors3];
idxTest = any([1:31]' == cellsTest, 2);
colorsTrain = colors(~idxTest, :);
colorsTest = colors(idxTest, :);

% Get two frequency model frequency indices array
idxFreq = nchoosek([1:69],2);

% Load model results
load('results\pipes_gpr_noModels.mat', 'Pipes_GPR');
load('results\pipes_linear_noModels.mat', 'Pipes_Linear')
load('results\pipes_rf_noModels.mat', 'Pipes_RF')

%% MAE vs. frequency for single freq models
% cv and test error versus model type
maeCrossValLinear = Pipes_Linear.maeCrossVal{3};
maeTestLinear = Pipes_Linear.maeTest{3};
maeCrossValGpr = Pipes_GPR.maeCrossVal{3};
maeTestGpr = Pipes_GPR.maeTest{3};
maeCrossValRF = Pipes_RF.maeCrossVal{3};
maeTestRF = Pipes_RF.maeTest{3};

% plot
figure; t=tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; grid on; box on; hold on;
plot(freq, maeCrossValLinear, ':', 'Color', colortriplet(1,:), 'LineWidth', 1.5, 'AlignVertexCenters', 'on')
plot(freq, maeCrossValGpr, ':', 'Color', colortriplet(2,:),  'LineWidth', 1.5, 'AlignVertexCenters', 'on')
plot(freq, maeCrossValRF, ':', 'Color', colortriplet(3,:),  'LineWidth', 1.5, 'AlignVertexCenters', 'on')
plot(freq, maeTestLinear, '-', 'Color', colortriplet(1,:),  'LineWidth', 1.5)
plot(freq, maeTestGpr, '-', 'Color', colortriplet(2,:),  'LineWidth', 1.5)
plot(freq, maeTestRF, '-', 'Color', colortriplet(3,:),  'LineWidth', 1.5)

% yline at minimum test error
mae_1freq_min = min([maeTestLinear; maeTestGpr; maeTestRF]);
yline(mae_1freq_min, '--k', 'LineWidth', 1.5)
annotation(gcf,'textbox',...
    [0.290598290598291 0.22 0.581196581196581 0.0833333333333333],...
    'String', sprintf("Minimum test MAE: %0.2g", mae_1freq_min),...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName', 'Arial');

% decorations
legend(...
    "Linear, CV", "GPR, CV", "RF, CV",...
    "Linear, Test", "GPR, Test", "RF, Test", ...
    'Location', 'northeast',...
    'NumColumns', 1,...
    'FontSize', 7,...
    'FontName', 'Arial')
set(gca, 'XScale', 'log'); ylim([0 0.15])
xticks([1e-2 1e0 1e2 1e4]); set(gca, 'XMinorGrid', 'Off')
xlabel('Frequency (Hz)');
ylabel('MAE')
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.177282051282051 0.825 0.0823333333333334 0.116666666666667],...
    'String',{'a'},...
    'FontSize',12,...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% set size
set(gcf, 'Units', 'inches', 'Position', [3,3,3.25,2.5])

% save
exportgraphics(gcf, 'figures/mae_vs_freq_singleFreqModels.eps', 'Resolution', 600)

%% MAE vs frequency for double freq models
% 2D contour plot of MAE test vs. frequency selection
% Get MAE vs frequency indices
maeTestLinear = Pipes_Linear.maeTest{4};
maeTestGpr = Pipes_GPR.maeTest{4};
maeTestRF = Pipes_RF.maeTest{4};
% colors
cmap = viridis(256); %cmap = [cmap; repmat(cmap(end,:), 300, 1)];
% levels for contours
levels = 0.018:0.002:0.05;

% Linear
z = nan(length(freq), length(freq));
for i = 1:size(idxFreq,1)
    z(idxFreq(i,1),idxFreq(i,2)) = maeTestLinear(i);
end
[x,y] = meshgrid(freq,freq);
figure; tiledlayout('flow'); nexttile;
contourf(y,x,z,levels); colormap(cmap);
cb = colorbar(); cb.Label.String = "MAE_{Test}"; cb.Layout.Tile = 'east';
cb.Label.Position = [-1.183333396911621,0.038602659295312,0];
set(gca, 'XScale', 'log', 'Yscale', 'log')
hold on; box on; grid on;
[~, idxBest] = min(maeTestLinear);
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
    'String',sprintf("Minimum MAE: %0.2g", min(maeTestLinear)),...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName', 'Arial');
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.19651282051282,0.808333333333333,0.082333333333333,0.116666666666667],...
    'String',{'b'},...
    'FontSize',12,...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');
exportgraphics(gcf, 'figures/mae_vs_freq_doubleFreq_linear.eps', 'Resolution', 600)

% Gpr
z = nan(length(freq), length(freq));
for i = 1:size(idxFreq,1)
    z(idxFreq(i,1),idxFreq(i,2)) = maeTestGpr(i);
end
[x,y] = meshgrid(freq,freq);
figure; tiledlayout('flow'); nexttile;
contourf(y,x,z,levels); colormap(cmap);
cb = colorbar(); cb.Label.String = "MAE_{Test}"; cb.Layout.Tile = 'east';
cb.Label.Position = [-1.183333396911621,0.035602659295312,0];
set(gca, 'XScale', 'log', 'Yscale', 'log')
hold on; box on; grid on;
[~, idxBest] = min(maeTestGpr);
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
    'String',sprintf("Minimum MAE: %0.2g", min(maeTestGpr)),...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName', 'Arial');
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.19651282051282,0.808333333333333,0.082333333333333,0.116666666666667],...
    'String',{'c'},...
    'FontSize',12,...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');
exportgraphics(gcf, 'figures/mae_vs_freq_doubleFreq_gpr.eps', 'Resolution', 600)

% RF
z = nan(length(freq), length(freq));
for i = 1:size(idxFreq,1)
    z(idxFreq(i,1),idxFreq(i,2)) = maeTestRF(i);
end
[x,y] = meshgrid(freq,freq);
figure; tiledlayout('flow'); nexttile;
contourf(y,x,z,levels); colormap(cmap);
cb = colorbar(); cb.Label.String = "MAE_{Test}"; cb.Layout.Tile = 'east';
cb.Label.Position = [-1.183333396911621,0.036602659295312,0];
set(gca, 'XScale', 'log', 'Yscale', 'log')
hold on; box on; grid on;
[~, idxBest] = min(maeTestRF);
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
    'String',sprintf("Minimum MAE: %0.2g", min(maeTestRF)),...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName', 'Arial');
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.19651282051282,0.808333333333333,0.082333333333333,0.116666666666667],...
    'String',{'d'},...
    'FontSize',12,...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');
exportgraphics(gcf, 'figures/mae_vs_freq_doubleFreq_rf.eps', 'Resolution', 600)

%% Heatmap of MAEs
plotMaeHeatmap(Pipes_Linear, "Linear")
plotMaeHeatmap(Pipes_GPR, "GPR");
plotMaeHeatmap(Pipes_RF, "RF");

%% Z vs q correlation plot with vertical lines for selected features from single freq, double freq, fscorr, fssisso
load('data\Data_Denso2021.mat', 'DataFormatted', 'Data2Formatted')
Data = combineDataTables(DataFormatted, Data2Formatted);
clearvars DataFormatted Data2Formatted
% Load full data files with models and all for this plot
load('results\pipes_gpr.mat', 'Pipes_GPR');
load('results\pipes_linear.mat', 'Pipes_Linear')
load('results\pipes_rf', 'Pipes_RF')

% plot
plotFeatureSelections(Pipes_Linear, "Linear", Data, freq)
plotFeatureSelections(Pipes_GPR, "GPR", Data, freq)
plotFeatureSelections(Pipes_RF, "RF", Data, freq)

clearvars Data Pipes_Linear Pipes_GPR Pipes_RF

%% MAE train on x-axis, MAE CV and MAE test on y axis, all models
% % % Load model results
% % load('results\pipes_gpr_noModels.mat', 'Pipes_GPR');
% % load('results\pipes_linear_noModels.mat', 'Pipes_Linear')
% % load('results\pipes_rf_noModels.mat', 'Pipes_RF')

[maeTrain_linear, maeCV_linear, maeTest_linear] = unwrapMAEs(Pipes_Linear);
[maeTrain_gpr, maeCV_gpr, maeTest_gpr] = unwrapMAEs(Pipes_GPR);
[maeTrain_rf, maeCV_rf, maeTest_rf] = unwrapMAEs(Pipes_RF);

maeTrain_baseline = Pipes_RF.maeTrain{1,:};
maeTest_baseline  = Pipes_RF.maeTest{1,:};
maeCV_baseline    = Pipes_RF.maeCrossVal{1,:};

% plot using 2D-histogram so point density is also visualized
nbins = 40;
xedges = linspace(0, 0.075, nbins);
yedges = linspace(0, 0.2, nbins);

figure; t=tiledlayout(2, 3, 'Padding','compact','TileSpacing','compact');
% include text arrow to extreme outliers
% mae test vs mae train, compare v baseline, ax1-linear, ax2-gpr, ax3-rf
nexttile; box on; grid on; hold on;
histogram2(gca, maeTrain_linear, maeTest_linear, xedges, yedges, 'DisplayStyle', 'Tile', 'ShowEmptyBins', 'On', 'EdgeColor', 'none');
colormap(gca, [0.8,0.8,0.8; brewermap('RdPu', 255)]);
% plot(maeTrain_baseline, maeTest_baseline, 'dk', 'LineWidth', 2);
xline(maeTrain_baseline, '--k', 'LineWidth', 1); yline(maeTest_baseline, '--k', 'LineWidth', 1)
ylim([0 0.2]); xlim([0 0.075]); 
xlabel('MAE_{Train}'); ylabel('MAE_{Test}');
title('Linear', 'FontWeight', 'normal')
set(gca, 'FontName', 'Arial')

nexttile; box on; grid on; hold on;
histogram2(gca, maeTrain_gpr, maeTest_gpr, xedges, yedges, 'DisplayStyle', 'Tile', 'ShowEmptyBins', 'On', 'EdgeColor', 'none');
colormap(gca, [0.8,0.8,0.8; brewermap('Blues', 255)]);
% plot(maeTrain_baseline, maeTest_baseline, 'dk', 'LineWidth', 2)
xline(maeTrain_baseline, '--k', 'LineWidth', 1); yline(maeTest_baseline, '--k', 'LineWidth', 1)
ylim([0 0.2]); xlim([0 0.075]);
xlabel('MAE_{Train}'); ylabel('MAE_{Test}');
title('GPR', 'FontWeight', 'normal')
set(gca, 'FontName', 'Arial')

nexttile; box on; grid on; hold on;
histogram2(gca, maeTrain_rf, maeTest_rf, xedges, yedges, 'DisplayStyle', 'Tile', 'ShowEmptyBins', 'On', 'EdgeColor', 'none');
colormap(gca, [0.8,0.8,0.8; brewermap('Greens', 255)]);
% plot(maeTrain_baseline, maeTest_baseline, 'dk', 'LineWidth', 2)
xline(maeTrain_baseline, '--k', 'LineWidth', 1); yline(maeTest_baseline, '--k', 'LineWidth', 1)
ylim([0 0.2]); xlim([0 0.075]);
xlabel('MAE_{Train}'); ylabel('MAE_{Test}');
title('RF', 'FontWeight', 'normal')
set(gca, 'FontName', 'Arial')

cb = colorbar();
cb.Layout.Tile = 'east';
cb.Label.String = 'Counts';
cb.Label.Position = [-0.080833256244659,125.7824167907238,0];
cb.Label.FontSize = 10;
cb.Label.FontName = 'Arial';

% mae CV vs mae train, compare v baseline, ax1-linear, ax2-gpr, ax3-rf
nexttile; box on; grid on; hold on;
histogram2(gca, maeTrain_linear, maeCV_linear, xedges, yedges, 'DisplayStyle', 'Tile', 'ShowEmptyBins', 'On', 'EdgeColor', 'none');
colormap(gca, [0.8,0.8,0.8; brewermap('RdPu', 255)]);
% plot(maeTrain_baseline, maeCV_baseline, 'dk', 'LineWidth', 2)
xline(maeTrain_baseline, '--k', 'LineWidth', 1); yline(maeCV_baseline, '--k', 'LineWidth', 1)
ylim([0 0.2]); xlim([0 0.075]);
xlabel('MAE_{Train}'); ylabel('MAE_{CV}');
set(gca, 'FontName', 'Arial')
nexttile; box on; grid on; hold on;
histogram2(gca, maeTrain_gpr, maeCV_gpr, xedges, yedges, 'DisplayStyle', 'Tile', 'ShowEmptyBins', 'On', 'EdgeColor', 'none');
colormap(gca, [0.8,0.8,0.8; brewermap('Blues', 255)]);
% plot(maeTrain_baseline, maeCV_baseline, 'dk', 'LineWidth', 2)
xline(maeTrain_baseline, '--k', 'LineWidth', 1); yline(maeCV_baseline, '--k', 'LineWidth', 1)
ylim([0 0.2]); xlim([0 0.075]);
xlabel('MAE_{Train}'); ylabel('MAE_{CV}');
nexttile; box on; grid on; hold on;
set(gca, 'FontName', 'Arial')
histogram2(gca, maeTrain_rf, maeCV_rf, xedges, yedges, 'DisplayStyle', 'Tile', 'ShowEmptyBins', 'On', 'EdgeColor', 'none');
colormap(gca, [0.8,0.8,0.8; brewermap('Greens', 255)]);
% plot(maeTrain_baseline, maeCV_baseline, 'dk', 'LineWidth', 2)
xline(maeTrain_baseline, '--k', 'LineWidth', 1); yline(maeCV_baseline, '--k', 'LineWidth', 1)
ylim([0 0.2]); xlim([0 0.075]);
xlabel('MAE_{Train}'); ylabel('MAE_{CV}');
set(gca, 'FontName', 'Arial')

set(gcf, 'Units', 'inches', 'Position', [3,1,6.5,3.5])

annotation(gcf,'textbox',...
    [0.0854017094017094 0.850198412698413 0.0321196581196581 0.0773809523809526],...
    'String',{'a'},...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
annotation(gcf,'textbox',...
    [0.384547008547008 0.850198412698413 0.0321196581196578 0.0773809523809526],...
    'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
annotation(gcf,'textbox',...
    [0.683692307692307 0.850198412698413 0.0321196581196581 0.0773809523809526],...
    'String','c',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
annotation(gcf,'textbox',...
    [0.0875384615384615 0.374007936507936 0.0321196581196582 0.0773809523809524],...
    'String','d',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
annotation(gcf,'textbox',...
    [0.38668376068376 0.374007936507936 0.0321196581196577 0.0773809523809524],...
    'String','e',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
annotation(gcf,'textbox',...
    [0.685829059829059 0.374007936507936 0.0321196581196581 0.0773809523809524],...
    'String','f',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');

% exportgraphics(gcf, 'figures/mae_histogram2_allmodels.tif', 'Resolution', 600);
exportgraphics(gcf, 'figures/mae_histogram2_allmodels.eps', 'Resolution', 600);


%% Show predictions, partial dependence of best three models
%{
Report detailed results of the best models found during the model
optimization process (pipeline search and hyperparameter optimization).
%}
% clean up
clearvars -except colortriplet; clc; close all;
% freq vector
load('data\Data_Denso2021.mat', 'Data')
freq = Data.Freq(1,:); clearvars Data

% Load data
load('data\Data_Denso2021.mat', 'DataFormatted', 'Data2Formatted')
Data = combineDataTables(DataFormatted, Data2Formatted);
% Remove series 43 and 44 (cells measured at BOL and not aged)
Data(Data.seriesIdx == 43, :) = [];
Data(Data.seriesIdx == 44, :) = [];
clearvars DataFormatted Data2Formatted

% Pull out X and Y data tables. X variables are any Zreal, Zimag, Zmag, and
% Zphz data points. Y variable is the relative discharge capacity, q.
X = Data(:, 5:end); Y = Data(:,2); seriesIdx = Data{:, 1};
% Data from cells running a WLTP drive cycle, and some from the aging study
% are used as test data.
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data.seriesIdx == cellsTest,2);
data.Xtest = X(maskTest,:); data.Ytest = Y(maskTest,:); 
seriesIdxTest = seriesIdx(maskTest);
% Cross-validation and training are conducted on the same data split.
data.Xcv = X(~maskTest,:); data.Ycv = Y(~maskTest,:); 
seriesIdxCV = seriesIdx(~maskTest);
% Define a cross-validation scheme.
cvsplit = cvpartseries(seriesIdxCV, 'Leaveout');

% Linear_1A (all freq) (hyperparameter optimized) model
seq = {...
    @RegressionPipeline.normalizeZScore,...
    };
Model_Linear = RegressionPipeline(@fitrlinear,...
    "FeatureTransformationSequence", seq,...
    "ModelFuncOpts", {'Lambda', 0.00015343, 'Learner', 'svm', 'Regularization', 'ridge'});
[Model_Linear, PredTrain_Linear, ~, PredCV_Linear] = crossvalidate(Model_Linear, data.Xcv, data.Ycv, cvsplit, seriesIdxCV);
PredTest_Linear = predict(Model_Linear, data.Xtest, data.Ytest, seriesIdxTest);
% Plot predictions
plot_parity_all(PredTrain_Linear, PredCV_Linear, PredTest_Linear, "Linear_1A_HypOpt")

% For linear model, plot beta directly instead of partial dependence
% Plot versus correlation to aid interpretation
% indices for correlation
w = size(X,2)/4;
idxZreal = 1:w;
idxZimag = (w+1):(w*2);
idxZmag = (w*2+1):(w*3);
idxZphz = (w*3+1):(w*4);
% Beta values
beta = Model_Linear.Model.Beta .* 10^2;
beta_zreal = beta(1:69);
beta_zimag = beta((69+1):(69*2));
beta_zmag  = beta((69*2 + 1):(69*3));
beta_zphz  = beta((69*3 + 1):end);
% Plot
figure; t = tiledlayout('flow', 'Padding', 'compact', 'TileSpacing', 'compact');
ax1 = nexttile; hold on; box on; grid on; colororder([[0,0,0];colortriplet(2,:)]);
title('Z_{Real}', 'FontWeight', 'normal');
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 0.5);
plot(freq, beta_zreal, '-k', 'LineWidth', 1.5);
xline(freq(5),'--', 'Color', colortriplet(1,:), 'LineWidth', 1.5)
set(gca, 'XScale', 'log'); ylim([-1.5, 1.5]); yticks(-1.5:0.5:1.5)
xlabel('Frequency (Hz)', 'FontSize', 10); ylabel('\beta\cdot10^2' , 'FontSize', 10);
yyaxis right; 
plot(freq, corr(data.Ycv{:,:}, data.Xcv{:,idxZreal}), '-', 'Color', colortriplet(2,:), 'LineWidth', 1.5);
ylim([-1 1]);
set(gca, 'FontName', 'Arial')
ax2 = nexttile; hold on; box on; grid on; colororder([[0,0,0];colortriplet(2,:)]);
title('Z_{Imaginary}', 'FontWeight', 'normal');
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 0.5);
plot(freq, beta_zimag, '-k', 'LineWidth', 1.5);
set(gca, 'XScale', 'log'); ylim([-1.5, 1.5]); yticks(-1.5:0.5:1.5)
xlabel('Frequency (Hz)', 'FontSize', 10);
yyaxis right; 
plot(freq, corr(data.Ycv{:,:}, data.Xcv{:,idxZimag}), '-', 'Color', colortriplet(2,:), 'LineWidth', 1.5);
ylim([-1 1]);
set(gca, 'FontName', 'Arial')
ax3 = nexttile; hold on; box on; grid on; colororder([[0,0,0];colortriplet(2,:)]);
title('|Z|', 'FontWeight', 'normal');
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 0.5);
plot(freq, beta_zmag, '-k', 'LineWidth', 1.5);
set(gca, 'XScale', 'log'); ylim([-1.5, 1.5]); yticks(-1.5:0.5:1.5)
xlabel('Frequency (Hz)', 'FontSize', 10);
yyaxis right; 
plot(freq, corr(data.Ycv{:,:}, data.Xcv{:,idxZmag}), '-', 'Color', colortriplet(2,:), 'LineWidth', 1.5);
ylim([-1 1]);
set(gca, 'FontName', 'Arial')
ax4 = nexttile; hold on; box on; grid on; colororder([[0,0,0];colortriplet(2,:)]);
title('\angleZ', 'FontWeight', 'normal');
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 0.5);
plot(freq, beta_zphz, '-k', 'LineWidth', 1.5);
xline(freq(222-69*3),'--', 'Color', colortriplet(1,:), 'LineWidth', 1.5)
set(gca, 'XScale', 'log'); ylim([-1.5, 1.5]); yticks(-1.5:0.5:1.5)
xlabel('Frequency (Hz)', 'FontSize', 10);
yyaxis right; 
plot(freq, corr(data.Ycv{:,:}, data.Xcv{:,idxZphz}), '-', 'Color', colortriplet(2,:), 'LineWidth', 1.5);
ylim([-1 1]);
set(gca, 'FontName', 'Arial')
ylabel({'Correlation to rel.','discharge capacity'})

set(gcf, 'Units', 'inches', 'Position', [2,2,6.5,1.9])

annotation(gcf,'textbox',...
    [0.0779230769230769 0.724285714285714 0.0374615384615385 0.131868131868132],...
    'String',{'a'},...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.305487179487179 0.724285714285714 0.0374615384615384 0.131868131868132],...
    'String','b',...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.534653846153846 0.724285714285714 0.0374615384615384 0.131868131868132],...
    'String','c',...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.75901282051282 0.724285714285714 0.0374615384615384 0.131868131868132],...
    'String','d',...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');

exportgraphics(gcf, 'figures/model_linear_best_beta.eps', 'Resolution', 600)
%%
% Plot 2D partial dependence of the most important magnitude and most
% important phase feature
% % % [a, idx] = sort(abs(beta), 'descend'); % [5, 220]
plot_2D_PDP(Model_Linear, PredTrain_Linear, [5,220])
fname = "Linear_1A_HypOpt";
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.047474358974358,0.845119047619047,0.071115384615385,0.131868131868133],...
    'String','e',...
    'FontName','Arial',...
    'FontSize', 12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
xlabel('Norm. Z_{Real} 1.3e+04 Hz')
ylabel('Norm. \angleZ 2e+03 Hz')

savefig(gcf, "figures/" + fname + "_2D_PDP")
exportgraphics(gcf, "figures/" + fname + "_2D_PDP.eps", 'Resolution', 600);

%%
% GPR_1C (best) model
seq = {...
    @RegressionPipeline.normalizeZScore,...
    @selectFrequency...
    };
hyp = {{}, {"idxFreq", [27,32]}};
Model_GPR = RegressionPipeline(@fitrgp,...
    "FeatureTransformationSequence", seq,...
    "FeatureTransformationFixedHyp", hyp);
[Model_GPR, PredTrain_GPR, ~, PredCV_GPR] = crossvalidate(Model_GPR, data.Xcv, data.Ycv, cvsplit, seriesIdxCV);
PredTest_GPR = predict(Model_GPR, data.Xtest, data.Ytest, seriesIdxTest);

% % % Plot predictions
% % plot_parity_all(PredTrain_GPR, PredCV_GPR, PredTest_GPR, "GPR_1E")
plot_1D_PDP(Model_GPR, "GPR_1E")
%%
plot_2D_PDP(Model_GPR, PredTrain_GPR, [2,7])
fname = "GPR_1E";
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.035,0.845119047619047,0.071115384615385,0.131868131868133],...
    'String','f',...
    'FontName','Arial',...
    'FontSize', 12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
xlabel('Norm. Z_{Real} 25 Hz')
ylabel('Norm. \angleZ 79 Hz')

savefig(gcf, "figures/" + fname + "_2D_PDP")
exportgraphics(gcf, "figures/" + fname + "_2D_PDP.eps", 'Resolution', 600);
%%
% RF_1C (double freq) (best) (hyperparameter optimized) (weighted) model
weights = evenlyWeightDataSeries(seriesIdxCV);
seq = {...
    @RegressionPipeline.normalizeZScore,...
    @selectFrequency...
    };
hyp = {{}, {"idxFreq", [19,43]}};
Model_RF = RegressionPipeline(@fitrensemble,...
    "FeatureTransformationSequence", seq,...
    "FeatureTransformationFixedHyp", hyp,...
    "ModelFuncOpts", {...
        'Weights', weights, ...
        'Method','Bag',...
        'OptimizeHyperparameters',...
            {'NumLearningCycles',...
             'MinLeafSize',...
             'MaxNumSplits',...
             'NumVariablesToSample'}...
             });
[Model_RF, PredTrain_RF, ~, PredCV_RF] = crossvalidate(Model_RF, data.Xcv, data.Ycv, cvsplit, seriesIdxCV);
PredTest_RF = predict(Model_RF, data.Xtest, data.Ytest, seriesIdxTest);
% Plot predictions
% % % plot_parity_all(PredTrain_RF, PredCV_RF, PredTest_RF, "RF_1C_HypOpt_Weighted")
plot_1D_PDP(Model_RF, "RF_1C_HypOpt_Weighted")
%%
plot_2D_PDP(Model_RF, PredTrain_RF, [2,5])
fname = "RF_1C_HypOpt_Weighted";
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.035,0.845119047619047,0.071115384615385,0.131868131868133],...
    'String','g',...
    'FontName','Arial',...
    'FontSize', 12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
xlabel('Norm. Z_{Real} 2 Hz')
ylabel('Norm. |Z| 5e+02 Hz')

savefig(gcf, "figures/" + fname + "_2D_PDP")
exportgraphics(gcf, "figures/" + fname + "_2D_PDP.eps", 'Resolution', 600);

%%
save('results\top_3_models.mat');

%% Plot distribution of predictions from top 3 models
plot_pred_pdfs(Data(maskTest,:),...
    [PredTest_Linear, PredTest_GPR, PredTest_RF],...
    'split', {'-10 \circC', '0 \circC, 10 \circC', '25 \circC'},... 
    {'Linear', 'GPR', 'RF'}, colortriplet, "best_models_predTest_pdfs")

% Split data by temp, but group middle temps
Data.split(Data.TdegC_EIS == -10) = 1;
Data.split(Data.TdegC_EIS == 0 | Data.TdegC_EIS == 10) = 2;
Data.split(Data.TdegC_EIS == 25) = 3;
% Colors
colors1 = brewermap('RdPu', 22);
colors1 = colors1(end-10:end,:);
colors2 = brewermap('Blues', 38);
colors2 = colors2(end-17:end,:);
colors3 = brewermap('Greens', 5);
colors3 = colors3(end-1:end,:);
colortriplet = [colors1(3,:); colors2(12,:); colors3(2,:)];
clearvars colors1 colors2 colors3

%% Plot predictions from 1 cell from best GPR model with confidence
% parity plot, color points by EIS temperature
seriesTest = unique(seriesIdxTest);
% seriesTest = seriesTest(1);

%{
SOMETHING IS WRONG HERE
first two series are the same (?)
no series goes below 90% capacity which isn't true
%}

colorsT = viridis(4);
tempsC = [-10, 0, 10, 25];
% for thisSeries = seriesTest' % 3rd test series looks the nicest
for thisSeries = seriesTest(3) 
    figure; hold on; box on; grid on;
    maskSeries = Data.seriesIdx == thisSeries;
    tempC_series = Data.TdegC_EIS(maskSeries);
    maskSeries = PredTest_GPR.seriesIdx == thisSeries;
    YSeries = PredTest_GPR.Y{maskSeries,:};
    YPredSeries = PredTest_GPR.YPred{maskSeries,1};
    YPredCISeries = PredTest_GPR.YPred{maskSeries,[3,4]};
    go = gobjects(4, 1);
    for iTemp = 1:length(tempsC)
        thisTemp = tempsC(iTemp);
        maskTemp = tempC_series == thisTemp;
        go(iTemp) = errorbar(YSeries(maskTemp), YPredSeries(maskTemp), ...
            abs(YPredSeries(maskTemp)-YPredCISeries(maskTemp,1)),...
            abs(YPredSeries(maskTemp)-YPredCISeries(maskTemp,2)),...
            '.', 'MarkerSize', 20, 'Color', colorsT(iTemp,:), 'LineWidth', 1);
    end
    axis([0.65 1.1 0.65 1.1])
    axlims = axis;
    maxlim = max(abs(axlims));
    plot([-maxlim, maxlim], [-maxlim, maxlim], '--k', 'LineWidth', 1.5)
    axis(axlims);
    legend(go, {'-10 \circC', '0 \circC', '10 \circC', '25 \circC'}, 'Location', 'northoutside', 'NumColumns', 2)
    
    xlabel('Actual rel. discharge capacity')
    ylabel('Predicted rel. capacity')
    
    set(gcf, 'Units', 'inches', 'Position', [2, 2, 3.25, 3])
end

exportgraphics(gcf, 'figures/model_gpr_best_predTestCI.tif', 'Resolution', 600)
exportgraphics(gcf, 'figures/model_gpr_best_predTestCI.eps', 'Resolution', 600)

%% Performance of ensemble model of the best 3 models
YPred_Train_Ensemble = mean([PredTrain_Linear.YPred{:,1},...
    PredTrain_GPR.YPred{:,1},...
    PredTrain_RF.YPred{:,1}], 2);
YPred_CV_Ensemble = mean([PredCV_Linear.YPred{:,1},...
    PredCV_GPR.YPred{:,1},...
    PredCV_RF.YPred{:,1}], 2);
YPred_Test_Ensemble = mean([PredTest_Linear.YPred{:,1},...
    PredTest_GPR.YPred{:,1},...
    PredTest_RF.YPred{:,1}], 2);
% Create Prediction objects to do all the hard stuff for us
PredTrain_Ensemble = Prediction(X(~maskTest,:), YPred_Train_Ensemble, Y(~maskTest,:), seriesIdxCV);
PredCV_Ensemble = Prediction(X(~maskTest,:), YPred_CV_Ensemble, Y(~maskTest,:), seriesIdxCV);
PredTest_Ensemble = Prediction(X(maskTest,:), YPred_Test_Ensemble, Y(maskTest,:), seriesIdxTest);

% Plot predictions
plot_parity_all(PredTrain_Ensemble, PredCV_Ensemble, PredTest_Ensemble, "Ensemble")

%% Plot of MAE (yyaxis left) and maxE (yyaxis right) for best 3 models and ensemble
maeTrain = [...
    PredTrain_Linear.FitStats.mae, ...
    PredTrain_GPR.FitStats.mae, ...
    PredTrain_RF.FitStats.mae, ...
    PredTrain_Ensemble.FitStats.mae...
    ];
maeCV = [...
    PredCV_Linear.FitStats.mae, ...
    PredCV_GPR.FitStats.mae, ...
    PredCV_RF.FitStats.mae, ...
    PredCV_Ensemble.FitStats.mae...
    ];
maeTest = [...
    PredTest_Linear.FitStats.mae, ...
    PredTest_GPR.FitStats.mae, ...
    PredTest_RF.FitStats.mae, ...
    PredTest_Ensemble.FitStats.mae...
    ];
maxaeTrain = [...
    PredTrain_Linear.FitStats.maxae, ...
    PredTrain_GPR.FitStats.maxae, ...
    PredTrain_RF.FitStats.maxae, ...
    PredTrain_Ensemble.FitStats.maxae...
    ];
maxaeCV = [...
    PredCV_Linear.FitStats.maxae, ...
    PredCV_GPR.FitStats.maxae, ...
    PredCV_RF.FitStats.maxae, ...
    PredCV_Ensemble.FitStats.maxae...
    ];
maxaeTest = [...
    PredTest_Linear.FitStats.maxae, ...
    PredTest_GPR.FitStats.maxae, ...
    PredTest_RF.FitStats.maxae, ...
    PredTest_Ensemble.FitStats.maxae...
    ];

figure; tiledlayout('flow','TileSpacing','compact','Padding','compact');
xstrs = categorical(["Linear","GPR","RF","Ensemble"]);
nexttile; hold on; box on; grid on; colororder({'k','k'})
ms = 8; lw = 1.5;
plot(xstrs, maeTrain, 'o', 'Color', colortriplet(1,:), 'MarkerSize', ms, 'LineWidth', lw)
plot(xstrs, maeCV, 's', 'Color', colortriplet(2,:), 'MarkerSize', ms, 'LineWidth', lw)
plot(xstrs, maeTest, 'd', 'Color', colortriplet(3,:), 'MarkerSize', ms, 'LineWidth', lw)
lgd = legend('Train','Cross-validation','Test');
lgd.Layout.Tile = 'north'; lgd.NumColumns = 3;
ylabel('Mean absolute error')
ylim([0 Inf])
nexttile; hold on; box on; grid on;colororder({'k','k'})
plot(xstrs, maxaeTrain, 'o', 'Color', colortriplet(1,:), 'MarkerSize', ms, 'LineWidth', lw)
plot(xstrs, maxaeCV, 's', 'Color', colortriplet(2,:), 'MarkerSize', ms, 'LineWidth', lw)
plot(xstrs, maxaeTest, 'd', 'Color', colortriplet(3,:), 'MarkerSize', ms, 'LineWidth', lw)
ylabel('Max absolute error')
ylim([0 Inf])

set(gcf, 'Units', 'inches', 'Position', [2, 2, 6.5, 2.25])

exportgraphics(gcf, 'figures/best_models_err_summary.tif', 'Resolution', 600)
exportgraphics(gcf, 'figures/best_models_err_summary.eps', 'Resolution', 600)


%% Helper methods
function plot_parity_all(PredTrain, PredCV, PredTest, fname)
% line colors
cellsTest = [7,10,13,17,24,30,31];
colors1 = brewermap('RdPu', 22);
colors1 = colors1(end-10:end,:);
colors2 = brewermap('Blues', 38);
colors2 = colors2(end-17:end,:);
colors3 = brewermap('Greens', 5);
colors3 = colors3(end-1:end,:);
colors = [colors1;colors2;colors3];
idxTest = any([1:31]' == cellsTest, 2);
colorsTrain = colors(~idxTest, :);
colorsTest = colors(idxTest, :);
% plot
figure; t=tiledlayout(1,3,'Padding', 'compact', 'TileSpacing', 'compact');
ax1 = nexttile; hold on; box on; grid on;
plotyy(PredTrain, 'k.', 'ax', ax1, 'SeriesColors', colorsTrain, 'MarkerSize', 8)
xlabel('Actual rel. capacity'); 
ylabel("Predicted rel. capacity");
set(gca, 'FontName', 'Arial')
% axis square; axis equal
ax2 = nexttile; hold on; box on; grid on;
plotyy(PredCV, 'k.', 'ax', ax2, 'SeriesColors', colorsTrain, 'MarkerSize', 8)
xlabel('Actual rel. discharge capacity'); 
ylabel("Predicted rel. capacity");
set(gca, 'FontName', 'Arial')
% axis square; axis equal
ax3 = nexttile; hold on; box on; grid on;
plotyy(PredTest, 'k.', 'ax', ax3, 'SeriesColors', colorsTest, 'MarkerSize', 8)
xlabel('Actual rel. discharge capacity'); 
ylabel("Predicted rel. capacity");
set(gca, 'FontName', 'Arial')
% axis square; axis equal
linkaxes([ax1, ax2, ax3])
set(gcf, 'Units', 'inches', 'Position', [2,2,6.5,2])
annotation(gcf,'textbox',...
    [0.079525641025641 0.807291666666667 0.045474358974359 0.125],...
    'String',{'a'},...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
annotation(gcf,'textbox',...
    [0.400038461538461 0.807291666666667 0.0454743589743589 0.125],'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
annotation(gcf,'textbox',...
    [0.715743589743589 0.807291666666667 0.0454743589743589 0.125],'String','c',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');

% exportgraphics(gcf, "figures/" + fname + "_parity.tif", 'Resolution', 600);
exportgraphics(gcf, "figures/" + fname + "_parity.eps", 'Resolution', 600);
end

function plot_1D_PDP(Model, fname)
features = Model.FeatureVars;
nFeatures = length(features);
ax = [];
figure; tiledlayout('flow', 'Padding', 'compact', 'TileSpacing', 'compact');
for i = 1:nFeatures
    ax_ = nexttile; box on;
    plotPartialDependence(Model.Model, i, 'Conditional', 'absolute')
    xlabel(strrep(features(i),'_',' '))
    ylabel('Rel. discharge capacity')
    ax = [ax,ax_];
end
linkaxes(ax, 'y')
set(gcf, 'Units', 'inches', 'Position', [1,1,8,6]);
% 
% exportgraphics(gcf, "figures/" + fname + "_1D_PDP.tif", 'Resolution', 600);
% exportgraphics(gcf, "figures/" + fname + "_1D_PDP.eps", 'Resolution', 600);
end

function plot_2D_PDP(Model, PTrain, idxFeatures)
%rf
features = Model.FeatureVars;
M = Model.Model;
if isa(M, 'RegressionLinear')
    [pd, x, y] = partialDependence(M, idxFeatures, PTrain.X{:,:});
else
    [pd, x, y] = partialDependence(M, idxFeatures);
end
% 2D plot
figure;
contourf(x, y, pd); colormap(plasma(256)); hold on; grid on; box on;
% scatter of feature values
scatter(PTrain.X{:,idxFeatures(1)}, PTrain.X{:,idxFeatures(2)}, 40, '.k')
% decorations
cb = colorbar(); cb.Label.String = "Rel. discharge capacity";
xlabel(strrep(features(idxFeatures(1)),'_',' '));
ylabel(strrep(features(idxFeatures(2)),'_',' '));
axis square;
set(gcf, 'Units', 'inches', 'Position', [2,2,3.25,2.5]);
end

function plot_pred_pdfs(Data, PredsTest, splitVar, splitTitles, modelNames, colors, fname)
z = Data.(splitVar);
z_unique = unique(z);

figure; t=tiledlayout(1,length(z_unique)+1,...
    'Padding','compact','TileSpacing','compact');

q = linspace(0.7, 1.05);
lw = 1.5;

% All data
nexttile; hold on; box on;
% ground truth
Y = PredsTest(1).Y{:,:};
pd = fitdist(Y, 'kernel');
plot(q, pdf(pd,q), '-k', 'LineWidth', lw);
% predictions
for i = 1:length(PredsTest)
    pd = fitdist(PredsTest(i).YPred{:,1}, 'kernel');
    plot(q, pdf(pd,q), '-', 'Color', colors(i,:), 'LineWidth', lw);
end
xlabel('Rel. capacity');
title('All data', 'FontWeight', 'normal')
lgd = legend(['Actual', modelNames], 'NumColumns', length(PredsTest)+1); lgd.Layout.Tile = 'north';
% by temp
for i = 1:length(z_unique)
    nexttile; hold on; box on;
    thisZ = z_unique(i);
    maskZ = z == thisZ;
    pd = fitdist(Y(maskZ), 'kernel');
    plot(q, pdf(pd,q), '-k', 'LineWidth', lw);
    % predictions
    for ii = 1:length(PredsTest)
        pd = fitdist(PredsTest(ii).YPred{maskZ,1}, 'kernel');
        plot(q, pdf(pd,q), '-', 'Color', colors(ii,:), 'LineWidth', lw);
    end
    xlabel('Rel. capacity');
    title(splitTitles{i}, 'FontWeight', 'normal');
end
ylabel(t, 'Density');
set(gcf, 'Units', 'inches', 'Position', [2,2,6.5,2]);

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

exportgraphics(gcf, "figures/" + fname + ".tif", 'Resolution', 600);
exportgraphics(gcf, "figures/" + fname + ".eps", 'Resolution', 600);
end

function plotMaeHeatmap(Pipe, type)
if isempty(Pipe.PTrain{7})
    Pipe = [Pipe(1:6,:); Pipe(8:end,:)];
end
maeTrain = zeros(1,height(Pipe));
maeCrossVal = zeros(1,height(Pipe));
maeTest = zeros(1,height(Pipe));
for i = 1:height(Pipe)
    errTrain = Pipe.maeTrain{i};
    errCV = Pipe.maeCrossVal{i};
    errTest = Pipe.maeTest{i};
    [~, idxMinTest] = min(errTest);
    maeTrain(i) = errTrain(idxMinTest);
    maeCrossVal(i) = errCV(idxMinTest);
    maeTest(i) = errTest(idxMinTest);
end
ylabels = ["MAE_{Train}","MAE_{CV}","MAE_{Test}"];
xlabels = strrep(Pipe.name, "_", " ");
xlabels = strrep(xlabels, "Models", "Model");
xlabels = strrep(xlabels, "Model", "Pipeline");
% colors - accentuate low errors. Maps are linear, index logarithmically
idx = round(logspace(log10(1), log10(256), 256)); 
% idx = 1:256;
idx = fliplr(257-idx);
blues = flipud(brewermap('Blues',256)); blues = blues(idx,:);
greens = flipud(brewermap('Greens',256)); greens = greens(idx,:);
purples = flipud(brewermap('RdPu',256)); purples = purples(idx,:);
% color between second lowest and second highest values (prevents an awful
% map with extremely low or high train/test errors)
maeTrainSort = sort(maeTrain);
maeCVSort = sort(maeCrossVal);
maeTestSort = sort(maeTest);
% plot
figure; heatmap(xlabels, ylabels(1), maeTrain,...
    'Colormap', blues,...
    'CellLabelFormat', '%0.2g',...
    'ColorbarVisible', 'off',...
    'ColorLimits', [maeTrainSort(2), maeTrainSort(end-1)])
set(gcf, 'Units', 'inches', 'Position', [3.5,5,7.2,1])
set(gca, 'FontName', 'Arial')
% exportgraphics(gcf, "figures/heatmap_train_" + type + ".tif", 'Resolution', 600)
exportgraphics(gcf, "figures/heatmap_train_" + type + ".eps", 'Resolution', 600)

figure; heatmap(xlabels, ylabels(2), maeCrossVal,...
    'Colormap', greens,...
    'CellLabelFormat', '%0.2g',...
    'ColorbarVisible', 'off',...
    'ColorLimits', [maeCVSort(2), maeCVSort(end-1)])
set(gcf, 'Units', 'inches', 'Position', [3.5,5,7.2,1])
set(gca, 'FontName', 'Arial')
% exportgraphics(gcf, "figures/heatmap_cv_" + type + ".tif", 'Resolution', 600)
exportgraphics(gcf, "figures/heatmap_cv_" + type + ".eps", 'Resolution', 600)

figure; heatmap(xlabels, ylabels(3), maeTest,...
    'Colormap', purples,...
    'CellLabelFormat', '%0.2g',...
    'ColorbarVisible', 'off',...
    'ColorLimits', [maeTestSort(2), maeTestSort(end-1)])
set(gcf, 'Units', 'inches', 'Position', [3.5,5,7.2,1])
set(gca, 'FontName', 'Arial')
% exportgraphics(gcf, "figures/heatmap_test_" + type + ".tif", 'Resolution', 600)
exportgraphics(gcf, "figures/heatmap_test_" + type + ".eps", 'Resolution', 600)

end

function plotFeatureSelections(Pipe, type, Data, freq)
X = Data(:, 5:end); Y = Data(:,2);
cellsTest = [7,10,13,17,24,30,31];
maskTest = any(Data.seriesIdx == cellsTest,2);
X = X(~maskTest,:); Y = Y(~maskTest,:);

% single freq
maeSingle = Pipe.maeTest{3};
[~,idxSingle] = min(maeSingle);
% double freq
maeDouble = Pipe.maeTest{4};
[~,idxDouble] = min(maeDouble);
idxFreq = Pipe.Model{4}.idxFreq;
idxDouble = idxFreq(idxDouble,:);
% correlation
maeCorr = Pipe.maeTest{5};
[~,idxCorr] = min(maeCorr);
idxCorr = Pipe.Model{5}.idxSelected{idxCorr};
idxCorrVar = ceil(idxCorr/length(freq));
idxCorrFreq = mod(idxCorr,length(freq))+1;
% SISSO
maeSISSO = Pipe.maeTest{6};
[~,idxSisso] = min(maeSISSO);
idxSisso = Pipe.Model{6}.idxSelected{idxSisso};
idxSissoVar = ceil(idxSisso/length(freq));
idxSissoFreq = mod(idxSisso,length(freq))+1;

% indices
w = size(X,2)/4;
idxZreal = 1:w;
idxZimag = (w+1):(w*2);
idxZmag = (w*2+1):(w*3);
idxZphz = (w*3+1):(w*4);

% colors
c = tab10(4);
% dotted line width
lw = 1.25;

figure; t = tiledlayout(2, 4, 'Padding', 'compact', 'TileSpacing', 'compact');
% ZReal
nexttile; hold on; box on; grid on;
plot(freq, corr(Y{:,:}, X{:,idxZreal}), '-k', 'LineWidth', 1);
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 0.5);
xlabel(t,'Frequency (Hz)', 'FontSize', 10);
ylabel(["Correlation with";"rel. discharge capacity"], 'FontSize', 10);
title("Z_{Real} (\Omega)", 'FontWeight', 'normal'); 
ylim([-1 1]); set(gca, 'XScale', 'log')
xticklabels([]); xlim([min(freq),max(freq)]);
%features
xline(freq(idxSingle), '-', 'Color', c(1,:), 'LineWidth', lw);
for i = 1:length(idxDouble)
    xline(freq(idxDouble(i)), '-', 'Color', c(2,:), 'LineWidth', lw);
end
if any(idxCorrVar == 1)
    idx = idxCorrFreq(idxCorrVar == 1);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(3,:), 'LineWidth', lw);
    end
end
if any(idxSissoVar == 1)
    idx = idxSissoFreq(idxSissoVar == 1);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(4,:), 'LineWidth', lw);
    end
end
set(gca, 'FontName', 'Arial')

% Zimag
nexttile; hold on; box on; grid on;
plot(freq, corr(Y{:,:}, X{:,idxZimag}), '-k', 'LineWidth', 1)
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 0.5);
title("Z_{Imaginary} (\Omega)", 'FontWeight', 'normal');
ylim([-1 1]); set(gca, 'XScale', 'log')
xticklabels([]); yticklabels([]); xlim([min(freq),max(freq)]);
xline(freq(idxSingle), '-', 'Color', c(1,:), 'LineWidth', lw);
for i = 1:length(idxDouble)
    xline(freq(idxDouble(i)), '-', 'Color', c(2,:), 'LineWidth', lw);
end
if any(idxCorrVar == 2)
    idx = idxCorrFreq(idxCorrVar == 2);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(3,:), 'LineWidth', lw);
    end
end
if any(idxSissoVar == 2)
    idx = idxSissoFreq(idxSissoVar == 2);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(4,:), 'LineWidth', lw);
    end
end
set(gca, 'FontName', 'Arial')

% Zmag
nexttile; hold on; box on; grid on;
plot(freq, corr(Y{:,:}, X{:,idxZmag}), '-k', 'LineWidth', 1)
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 0.5);
title("|Z| (\Omega)", 'FontWeight', 'normal');
ylim([-1 1]); set(gca, 'XScale', 'log')
xticklabels([]); yticklabels([]); xlim([min(freq),max(freq)]);
xline(freq(idxSingle), '-', 'Color', c(1,:), 'LineWidth', lw);
for i = 1:length(idxDouble)
    xline(freq(idxDouble(i)), '-', 'Color', c(2,:), 'LineWidth', lw);
end
if any(idxCorrVar == 3)
    idx = idxCorrFreq(idxCorrVar == 3);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(3,:), 'LineWidth', lw);
    end
end
if any(idxSissoVar == 3)
    idx = idxSissoFreq(idxSissoVar == 3);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(4,:), 'LineWidth', lw);
    end
end
set(gca, 'FontName', 'Arial')

% Zphz
nexttile; hold on; box on; grid on;
plot(freq, corr(Y{:,:}, X{:,idxZphz}), '-k', 'LineWidth', 1)
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 0.5);
title("\angleZ (\circ)", 'FontWeight', 'normal');
ylim([-1 1]); set(gca, 'XScale', 'log')
xticklabels([]); yticklabels([]); xlim([min(freq),max(freq)]);
xline(freq(idxSingle), '-', 'Color', c(1,:), 'LineWidth', lw);
for i = 1:length(idxDouble)
    xline(freq(idxDouble(i)), '-', 'Color', c(2,:), 'LineWidth', lw);
end
if any(idxCorrVar == 4)
    idx = idxCorrFreq(idxCorrVar == 4);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(3,:), 'LineWidth', lw);
    end
end
if any(idxSissoVar == 4)
    idx = idxSissoFreq(idxSissoVar == 4);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(4,:), 'LineWidth', lw);
    end
end
set(gca, 'FontName', 'Arial')

% features
y = categorical(["Single freq.","Double freq.","Corr. search", "SISSO"], 'Ordinal', true);
%zreal
nexttile; hold on; box on; grid on;
plot(freq(idxSingle), y(1), 'ok', 'MarkerFaceColor', c(1,:));
plot(freq(idxDouble), y(2), 'sk', 'MarkerFaceColor', c(2,:));
if any(idxCorrVar == 1)
    plot(freq(idxCorrFreq(idxCorrVar == 1)), y(3), '^k', 'MarkerFaceColor', c(3,:));
end
if any(idxSissoVar == 1)
    plot(freq(idxSissoFreq(idxSissoVar == 1)), y(4), '>k', 'MarkerFaceColor', c(4,:));
end
xlim([min(freq),max(freq)]); set(gca, 'XScale', 'log')
yline(y(1), '--', 'Color', c(1,:)); yline(y(2), '--', 'Color', c(2,:));
yline(y(3), '--', 'Color', c(3,:)); yline(y(4), '--', 'Color', c(4,:));
xline(freq(idxSingle), '-', 'Color', c(1,:), 'LineWidth', lw);
for i = 1:length(idxDouble)
    xline(freq(idxDouble(i)), '-', 'Color', c(2,:), 'LineWidth', lw);
end
if any(idxCorrVar == 1)
    idx = idxCorrFreq(idxCorrVar == 1);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(3,:), 'LineWidth', lw);
    end
end
if any(idxSissoVar == 1)
    idx = idxSissoFreq(idxSissoVar == 1);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(4,:), 'LineWidth', lw);
    end
end
xticks([0.01 0.1 1 10 100 1000 10000]);
xticklabels({'10^{-2}', '', '10^0', '', '10^2', '', '10^4'});
set(gca, 'XMinorGrid', 'off')
set(gca, 'FontName', 'Arial')

%zimag
nexttile; hold on; box on; grid on;
plot(freq(idxSingle), y(1), 'ok', 'MarkerFaceColor', c(1,:));
plot(freq(idxDouble), y(2), 'sk', 'MarkerFaceColor', c(2,:));
if any(idxCorrVar == 2)
    plot(freq(idxCorrFreq(idxCorrVar == 2)), y(3), '^k', 'MarkerFaceColor', c(3,:));
end
if any(idxSissoVar == 2)
    plot(freq(idxSissoFreq(idxSissoVar == 2)), y(4), '>k', 'MarkerFaceColor', c(4,:));
end
xlim([min(freq),max(freq)]); set(gca, 'XScale', 'log')
yticklabels([]);
yline(y(1), '--', 'Color', c(1,:)); yline(y(2), '--', 'Color', c(2,:));
yline(y(3), '--', 'Color', c(3,:)); yline(y(4), '--', 'Color', c(4,:));
xline(freq(idxSingle), '-', 'Color', c(1,:), 'LineWidth', lw);
for i = 1:length(idxDouble)
    xline(freq(idxDouble(i)), '-', 'Color', c(2,:), 'LineWidth', lw);
end
if any(idxCorrVar == 2)
    idx = idxCorrFreq(idxCorrVar == 2);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(3,:), 'LineWidth', lw);
    end
end
if any(idxSissoVar == 2)
    idx = idxSissoFreq(idxSissoVar == 2);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(4,:), 'LineWidth', lw);
    end
end
xticks([0.01 0.1 1 10 100 1000 10000]);
xticklabels({'10^{-2}', '', '10^0', '', '10^2', '', '10^4'});
set(gca, 'XMinorGrid', 'off')
set(gca, 'FontName', 'Arial')

%zmag
nexttile; hold on; box on; grid on;
plot(freq(idxSingle), y(1), 'ok', 'MarkerFaceColor', c(1,:));
plot(freq(idxDouble), y(2), 'sk', 'MarkerFaceColor', c(2,:));
if any(idxCorrVar == 3)
    plot(freq(idxCorrFreq(idxCorrVar == 3)), y(3), '^k', 'MarkerFaceColor', c(3,:));
end
if any(idxSissoVar == 3)
    plot(freq(idxSissoFreq(idxSissoVar == 3)), y(4), '>k', 'MarkerFaceColor', c(4,:));
end
xlim([min(freq),max(freq)]); set(gca, 'XScale', 'log')
yticklabels([]);
yline(y(1), '--', 'Color', c(1,:)); yline(y(2), '--', 'Color', c(2,:));
yline(y(3), '--', 'Color', c(3,:)); yline(y(4), '--', 'Color', c(4,:));
xline(freq(idxSingle), '-', 'Color', c(1,:), 'LineWidth', lw);
for i = 1:length(idxDouble)
    xline(freq(idxDouble(i)), '-', 'Color', c(2,:), 'LineWidth', lw);
end
if any(idxCorrVar == 3)
    idx = idxCorrFreq(idxCorrVar == 3);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(3,:), 'LineWidth', lw);
    end
end
if any(idxSissoVar == 3)
    idx = idxSissoFreq(idxSissoVar == 3);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(4,:), 'LineWidth', lw);
    end
end
xticks([0.01 0.1 1 10 100 1000 10000]);
xticklabels({'10^{-2}', '', '10^0', '', '10^2', '', '10^4'});
set(gca, 'XMinorGrid', 'off')
set(gca, 'FontName', 'Arial')

%zphz
nexttile; hold on; box on; grid on;
plot(freq(idxSingle), y(1), 'ok', 'MarkerFaceColor', c(1,:));
plot(freq(idxDouble), y(2), 'sk', 'MarkerFaceColor', c(2,:));
if any(idxCorrVar == 4)
    plot(freq(idxCorrFreq(idxCorrVar == 4)), y(3), '^k', 'MarkerFaceColor', c(3,:));
end
if any(idxSissoVar == 4)
    plot(freq(idxSissoFreq(idxSissoVar == 4)), y(4), '>k', 'MarkerFaceColor', c(4,:));
end
xlim([min(freq),max(freq)]); set(gca, 'XScale', 'log')
yticklabels([]);
yline(y(1), '--', 'Color', c(1,:)); yline(y(2), '--', 'Color', c(2,:));
yline(y(3), '--', 'Color', c(3,:)); yline(y(4), '--', 'Color', c(4,:));
xline(freq(idxSingle), '-', 'Color', c(1,:), 'LineWidth', lw);
for i = 1:length(idxDouble)
    xline(freq(idxDouble(i)), '-', 'Color', c(2,:), 'LineWidth', lw);
end
if any(idxCorrVar == 4)
    idx = idxCorrFreq(idxCorrVar == 4);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(3,:), 'LineWidth', lw);
    end
end
if any(idxSissoVar == 4)
    idx = idxSissoFreq(idxSissoVar == 4);
    for i = 1:length(idx)
        xline(freq(idx(i)), '-', 'Color', c(4,:), 'LineWidth', lw);
    end
end
xticks([0.01 0.1 1 10 100 1000 10000]);
xticklabels({'10^{-2}', '', '10^0', '', '10^2', '', '10^4'});
set(gca, 'XMinorGrid', 'off')
set(gca, 'FontName', 'Arial')

set(gcf, 'Units', 'inches', 'Position', [3,1,6.5,4])

% exportgraphics(gcf, "figures/selected_features_plot_" + type + ".tif", 'Resolution', 600)
exportgraphics(gcf, "figures/selected_features_plot_" + type + ".eps", 'Resolution', 600)
end

function Data = combineDataTables(DataFormatted, Data2Formatted)
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

function [maeTrain, maeCV, maeTest] = unwrapMAEs(Pipe)
mae = Pipe.maeTrain(:);
mae_temp = [];
for i = 1:length(mae)
    thisMAE = mae{i};
    mae_temp = [mae_temp; thisMAE];
end
maeTrain = mae_temp;

mae = Pipe.maeCrossVal(:);
mae_temp = [];
for i = 1:length(mae)
    thisMAE = mae{i};
    mae_temp = [mae_temp; thisMAE];
end
maeCV = mae_temp;

mae = Pipe.maeTest(:);
mae_temp = [];
for i = 1:length(mae)
    thisMAE = mae{i};
    mae_temp = [mae_temp; thisMAE];
end
maeTest = mae_temp;
end