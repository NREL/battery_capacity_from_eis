% Paul Gasper, NREL, 2/2022
% A place to make data exploration figures for the article.
clc; clear; close all;

% Load the data tables
load('data\Data_Denso2021.mat', 'Data', 'Data2')
% Filter out noisy interpolated data
idxKeep = filterInterpData(Data);
Data = Data(idxKeep, :);

% Colormaps file path
addpath(genpath('functions'))

%% Plot 1: capacity fade trends, EIS measurement locations denoted
% Use only 25C data, since -10/0/10/25C data share the same capacity
% measurements / EIS measurement dates
DataA = Data(Data.TdegC_EIS == 25, :);
% Training/CV data
cellsTest = [7,10,13,17,24,30,31];
maskTestA = any(DataA.seriesIdx == cellsTest,2);
DataA1 = DataA(~maskTestA, :);
% Test data
DataA2 = DataA(maskTestA, :);

% Colors: second half of reds for calendar aging, second half of blues for
% cycle aging, greens for WLTP cells
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

colortriplet = [colors1(3,:); colors2(12,:); colors3(2,:)];

PlotOpt = setPlotOpt(...
    'DataSeriesLabelVar', 'name',...
    'DataSeriesLabelFormat', 'none',...
    'Colorbar','off');
DataLineProp = setLineProp('-','Marker','.',...
    'LineWidth',1,'MarkerSize',12,...
    'Color', colorsTrain);
plotData(DataA1, 't', 'q', DataLineProp, PlotOpt)
ylabel('C/3 relative capacity (25\circC)')
% Denote EIS measurements
maskEIS = ~logical(DataA1.isInterpEIS);
plot(DataA1.t(maskEIS), DataA1.q(maskEIS), '+k', 'LineWidth', 2, 'MarkerSize', 8);
ylim([0.65 1.1]); xlim([0 500])
set(gcf, 'Units', 'inches', 'Position', [5.010416666666666,5.520833333333333,3.25,2.749999999999999]);
annotation(gcf,'textbox',...
    [0.82151282051282 0.821969696969697 0.0791282051282054 0.0984848484848488],...
    'String','a',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');

DataLineProp = setLineProp('-','Marker','.',...
    'LineWidth',1,'MarkerSize',12,...
    'Color', colorsTest);
plotData(DataA2, 't', 'q', DataLineProp, PlotOpt)
ylabel('C/3 relative capacity (25\circC)')
% Denote EIS measurements
maskEIS = ~logical(DataA2.isInterpEIS);
plot(DataA2.t(maskEIS), DataA2.q(maskEIS), '+k', 'LineWidth', 2, 'MarkerSize', 12);
ylim([0.65 1.1]); xlim([0 500])
set(gcf, 'Units', 'inches', 'Position', [8.375,5.520833333333333,3.25,2.749999999999999]);
annotation(gcf,'textbox',...
    [0.82151282051282 0.821969696969697 0.0791282051282054 0.0984848484848488],...
    'String','b',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');

% DC resistance plots
PlotOpt = setPlotOpt(...
    'DataSeriesLabelVar', 'name',...
    'DataSeriesLabelFormat', 'none',...
    'Colorbar','off');
DataLineProp = setLineProp('-','Marker','.',...
    'LineWidth',1,'MarkerSize',12,...
    'Color', colorsTrain);
plotData(DataA1, 't', 'rm10C', DataLineProp, PlotOpt)
ylabel({'Rel. DC pulse resistance';'(-10\circC, 50% SOC)'})
% Denote EIS measurements
maskEIS = ~logical(DataA1.isInterpEIS);
plot(DataA1.t(maskEIS), DataA1.rm10C(maskEIS), '+k', 'LineWidth', 2, 'MarkerSize', 8);
ylim([0.7 1.7]); xlim([0 500])
set(gcf, 'Units', 'inches', 'Position', [5,1.84375,3.375,2.75]);
annotation(gcf,'textbox',...
    [0.824599240265906 0.753787878787879 0.0791282051282054 0.0984848484848488],...
    'String','c',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');

DataLineProp = setLineProp('-','Marker','.',...
    'LineWidth',1,'MarkerSize',12,...
    'Color', colorsTest);
plotData(DataA2, 't', 'rm10C', DataLineProp, PlotOpt)
ylabel({'Rel. DC pulse resistance';'(-10\circC, 50% SOC)'})
% Denote EIS measurements
maskEIS = ~logical(DataA2.isInterpEIS);
plot(DataA2.t(maskEIS), DataA2.rm10C(maskEIS), '+k', 'LineWidth', 2, 'MarkerSize', 12);
ylim([0.7 1.7]); xlim([0 500])
set(gcf, 'Units', 'inches', 'Position', [8.364583333333332,1.854166666666667,3.375,2.729166666666667]);
annotation(gcf,'textbox',...
    [0.82151282051282 0.753787878787879 0.0791282051282054 0.0984848484848488],...
    'String','d',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
ax = gca; kids = ax.Children;
lgd = legend(kids([8,5,3,1]), {'Storage','Cycling','Drive cycle','EIS'}, 'Location', 'northwest');
title(lgd, 'Aging condition')

%%
% Just plot all data on same graph
% capacity v time
PlotOpt = setPlotOpt(...
    'DataSeriesLabelVar', 'name',...
    'DataSeriesLabelFormat', 'none',...
    'Colorbar','off');
DataLineProp = setLineProp('-','Marker','.',...
    'LineWidth',1.5,'MarkerSize',8,...
    'Color', colors);
plotData(DataA, 't', 'q', DataLineProp, PlotOpt)
ylabel('C/3 relative capacity (25\circC)')
set(gcf, 'Units', 'inches', 'Position', [8,2,4,3]);
ax = gca; kids = ax.Children;
lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'southwest');
title(lgd, 'Aging condition')

% resistance v time
plotData(DataA, 't', 'rm10C', DataLineProp, PlotOpt)
ylabel({'Rel. DC pulse resistance';'(-10\circC, 50% SOC)'})
set(gcf, 'Units', 'inches', 'Position', [8,2,4,3]);
ax = gca; kids = ax.Children;
lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northwest');
title(lgd, 'Aging condition')

% capacity v resistance
plotData(DataA, 'rm10C', 'q', DataLineProp, PlotOpt)
ylabel({'Rel. C/3 discharge';'capacity (25\circC)'})
xlabel({'Rel. DC pulse resistance';'(-10\circC, 50% SOC)'})
set(gcf, 'Units', 'inches', 'Position', [9.260416666666666,2.479166666666667,3.25,3]);
ax = gca; kids = ax.Children;
lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northeast');
title(lgd, 'Aging condition')

%% multi-part plot:
% a - q vs r
% b - q vs r, polyfit
% c - q vs dq/dr from polyfit
% d - q vs 0.01s r
% e - q vs 0.1s r
% f - q vs 10s r


% a
PlotOpt = setPlotOpt(...
    'DataSeriesLabelVar', 'name',...
    'DataSeriesLabelFormat', 'none',...
    'Colorbar','off');
DataLineProp = setLineProp('-','Marker','.',...
    'LineWidth',1.5,'MarkerSize',8,...
    'Color', colors);
plotData(DataA, 'rm10C', 'q', DataLineProp, PlotOpt)
ylabel({'Rel. C/3 discharge';'capacity (25\circC)'})
xlabel({'Rel. DC pulse resistance';'(-10\circC, 50% SOC)'})
set(gcf, 'Units', 'inches', 'Position', [9.260416666666666,2.479166666666667,3.25,3]);
ax = gca; kids = ax.Children;
lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northeast');
title(lgd, 'Aging condition')
annotation(gcf,'textbox',...
    [0.82151282051282 0.753787878787879 0.0791282051282054 0.0984848484848488],...
    'String','a',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');

% b
% fit poly model locally, plot fit
DataA.groupIdx = DataA.seriesIdx;
polymdl = ReducedOrderModel('polymdl', 'rm10C = a + b*q + c*(q^2) + d*(q^3)', {'a','b','c','d'});
[polymdl, fit] = polymdl.optimizeLocal(DataA, {'a','b','c','d'}, [0,0,0,0]);
PlotOpt = setPlotOpt(...
    'DataSeriesLabelVar', 'name',...
    'DataSeriesLabelFormat', 'none',...
    'Colorbar','off',...
    'ResidualsPlots', 'off',...
    'FitStatsBox', 'off');
DataLineProp = setLineProp('.',...
    'LineWidth',1.5,'MarkerSize',8,...
    'Color', colors, 'MarkerFaceColor', colors);
FitLineProp = setLineProp('-',...
    'LineWidth', 1.5, 'Color', colors);
plotFit(fit, DataA, 'q', DataLineProp, FitLineProp, PlotOpt)
xlabel({'Rel. C/3 discharge';'capacity (25\circC)'})
ylabel({'Rel. DC pulse resistance';'(-10\circC, 50% SOC)'})
set(gcf, 'Units', 'inches', 'Position', [9.260416666666666,2.479166666666667,3.25,3]);
% ax = gca; kids = ax.Children;
% lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northeast');
% title(lgd, 'Aging condition')
annotation(gcf,'textbox',...
    [0.82151282051282 0.753787878787879 0.0791282051282054 0.0984848484848488],...
    'String','b',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');

% c - dq/dr for each line as a function of q
p = fit.p;
uniqueSeries = unique(DataA.seriesIdx, 'stable');
figure; hold on; box on; grid on;
yline(0, '--k','LineWidth', 2)
for i = 1:length(uniqueSeries)
    if ~(i == 29 || i == 11)
        thisSeries = uniqueSeries(i);
        maskSeries = DataA.seriesIdx == thisSeries;
        DataThisSeries = DataA(maskSeries, :);
        q = linspace(min(DataThisSeries.q), max(DataThisSeries.q));
        drdq = p(i,2) + 2.*p(i,3).*q + 3.*p(i,4).*(q.^2);
        plot(q, drdq, '-', 'Color', colors(i,:), 'LineWidth', 1.5)
    end
end
xlabel({'Rel. C/3 discharge';'capacity (25\circC)'})
ylabel({'Slope of rel. resistance versus';'rel. capacity, dr/dq'})
% ylabel('$$ \frac{dq}{dr}', 'Interpreter', 'latex')
set(gcf, 'Units', 'inches', 'Position', [9.260416666666666,2.479166666666667,3.25,3]);
ylim([-10 10]); xlim([0.65 1])
annotation(gcf,'textbox',...
    [0.21574358974359 0.78472222222222 0.486179487179487 0.118055555555554],...
    'String','\downarrow resist. = \downarrow cap.',...
    'FontSize',12,...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'arrow',[0.303 0.303],...
    [0.599694444444444 0.8125]);
annotation(gcf,'arrow',[0.303 0.303],...
    [0.526777777777775 0.309027777777777]);
annotation(gcf,'textbox',...
    [0.206128205128205 0.17361111111111 0.486179487179487 0.118055555555554],...
    'String','\uparrow resist. = \downarrow cap.',...
    'FontSize',12,...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.82151282051282 0.8 0.0791282051282054 0.0984848484848488],...
    'String','c',...
    'FontSize',12,...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% d-f: DCIR at different time constants
load('data\Denso\DensoData.mat', 'DensoData');
DensoData = DensoData(~isnan(DensoData.r25C50soc10s), :);

PlotOpt = setPlotOpt(...
    'DataSeriesLabelVar', 'name',...
    'DataSeriesLabelFormat', 'none',...
    'Colorbar','off');
DataLineProp = setLineProp('-','Marker','.',...
    'LineWidth',1.5,'MarkerSize',8,...
    'Color', colors([1:25,27:end],:));
%0p01s
plotData(DensoData, 'rm10C50soc0p01s', 'q', DataLineProp, PlotOpt)
ylabel({'Rel. C/3 discharge';'capacity (25\circC)'})
xlabel({'Rel. DC pulse resistance';'(0-0.01 s, -10\circC, 50% SOC)'})
set(gcf, 'Units', 'inches', 'Position', [9.260416666666666,2.479166666666667,3.25,3]);
% ax = gca; kids = ax.Children;
% lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northeast');
% title(lgd, 'Aging condition')
annotation(gcf,'textbox',...
    [0.82151282051282 0.753787878787879 0.0791282051282054 0.0984848484848488],...
    'String','d',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
%0p1s
plotData(DensoData, 'rm10C50soc0p1s', 'q', DataLineProp, PlotOpt)
ylabel({'Rel. C/3 discharge';'capacity (25\circC)'})
xlabel({'Rel. DC pulse resistance';'(0.01-0.1 s, -10\circC, 50% SOC)'})
set(gcf, 'Units', 'inches', 'Position', [9.260416666666666,2.479166666666667,3.25,3]);
% ax = gca; kids = ax.Children;
% lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northeast');
% title(lgd, 'Aging condition')
annotation(gcf,'textbox',...
    [0.82151282051282 0.753787878787879 0.0791282051282054 0.0984848484848488],...
    'String','e',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
%10s
plotData(DensoData, 'rm10C50soc10s', 'q', DataLineProp, PlotOpt)
ylabel({'Rel. C/3 discharge';'capacity (25\circC)'})
xlabel({'Rel. DC pulse resistance';'(0.1-10 s, -10\circC, 50% SOC)'})
set(gcf, 'Units', 'inches', 'Position', [9.260416666666666,2.479166666666667,3.25,3]);
xticks([0,1,3,5,7]);
% ax = gca; kids = ax.Children;
% lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northeast');
% title(lgd, 'Aging condition')
annotation(gcf,'textbox',...
    [0.82151282051282 0.753787878787879 0.0791282051282054 0.0984848484848488],...
    'String','f',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');

%% voltage profiles
load('C:\Users\pgasper\Documents\GitHub\denso_eis_pipeline\data\Denso\DensoData.mat')
d = DensoData(280:302,:);
for i = 1:height(d)
    discharge = d.Discharge{i};
    dt = [0;diff(discharge.Time)];
    ah = cumsum(dt.*abs(discharge.Current))./3600;
    discharge.AmpHours = ah;
    d.Discharge{i} = discharge;
end
figure; hold on; box on; grid on;
c = colormap(plasma(height(d)));
for i = 1:height(d)
    discharge = d.Discharge{i};
    plot(discharge.AmpHours, discharge.Voltage, '-', 'Color', c(i,:), 'LineWidth', 1.5)
end
xlabel('Discharged capacity (Ah)');
ylabel('Voltage (V)');

set(gcf, 'Units', 'inches', 'Position', [8,2,4,3]);

%% Plot 2: EIS trends versus capacity
% For 1 storage, 1 cycling cell, plot EIS curves for -10C and 25C data
% Color by capacity (capacity map: 0.6:1) (colormap: cividis) (solid line w/ markers at frequency decades)
capacity = linspace(1, 0.6, 256);
colors = plasma(256);

% indices of frequency decades
freq = Data.Freq(1,:);
idxDecades = log10(freq) == round(log10(freq));

% Storage cell: cell 1
DataEIS1 = Data(Data.seriesIdx == 1, :);
DataEIS1_m10C = DataEIS1(DataEIS1.TdegC_EIS == -10, :);
DataEIS1_25C = DataEIS1(DataEIS1.TdegC_EIS == 25, :);

figure; t = tiledlayout('flow', 'Padding', 'compact', 'TileSpacing', 'compact');
% Plot -10C nyquist
nexttile; D = DataEIS1_m10C; hold on; box on; grid on;
for i = 1:height(D)
    [~, idxColor] = min(abs(capacity - D.q(i)));
    if D.isInterpEIS(i)
        % interpolated EIS
        c = colors(idxColor,:);
        plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, ':', 'Color', c, 'LineWidth', 1)
        plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c)
    else
        % raw EIS
        c = colors(idxColor,:);
        plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
        plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
    end
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0 0.008].*1e4); ylim([-0.005 0.003].*1e4); axis('square')
% Plot 25C nyquist
nexttile; D = DataEIS1_25C; hold on; box on; grid on;
for i = 1:height(D)
    [~, idxColor] = min(abs(capacity - D.q(i)));
    if D.isInterpEIS(i)
        % interpolated EIS
        c = colors(idxColor,:);
        plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, ':', 'Color', c, 'LineWidth', 1)
        plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c)
    else
        % raw EIS
        c = colors(idxColor,:);
        plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
        plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
    end
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0.0006 0.0012].*1e4); ylim([-0.00045 0.00015].*1e4); axis('square')

% Cycling cell: cell 16
DataEIS2 = Data(Data.seriesIdx == 16, :);
DataEIS2_m10C = DataEIS2(DataEIS2.TdegC_EIS == -10, :);
DataEIS2_25C = DataEIS2(DataEIS2.TdegC_EIS == 25, :);

% Plot -10C nyquist
nexttile; D = DataEIS2_m10C; hold on; box on; grid on;
for i = 1:height(D)
    [~, idxColor] = min(abs(capacity - D.q(i)));
    if D.isInterpEIS(i)
        % interpolated EIS
        c = colors(idxColor,:);
        plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, ':', 'Color', c, 'LineWidth', 1)
        plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c)
    else
        % raw EIS
        c = colors(idxColor,:);
        plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
        plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
    end
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0 0.008].*1e4); ylim([-0.005 0.003].*1e4); axis('square')
% Plot 25C nyquist
nexttile; D = DataEIS2_25C; hold on; box on; grid on;
for i = 1:height(D)
    [~, idxColor] = min(abs(capacity - D.q(i)));
    if D.isInterpEIS(i)
        % interpolated EIS
        c = colors(idxColor,:);
        plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, ':', 'Color', c, 'LineWidth', 1)
        plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c)
    else
        % raw EIS
        c = colors(idxColor,:);
        plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
        plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
    end
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0.0006 0.0012].*1e4); ylim([-0.00045 0.00015].*1e4); axis('square')

tickLabels = round(linspace(floor(min(capacity)*100), 100, 5));
ticklabels = compose("%d%%", tickLabels);

c = colorbar('Colormap', flipud(colors), 'Limits', [0 1],...
    'Ticks', [0 0.25 0.5 0.75 1], 'TickLabels', ticklabels);
c.Label.String = 'Rel. discharge capacity';
c.Label.Position = [-1.065833330154419,0.494624137878418,0];
c.TickLength = 0.02;
c.Layout.Tile = 'east';

set(gcf, 'Units', 'inches', 'Position', [4.697916666666666,6.65625,7.90625,1.75]);

annotation(gcf,'textbox',...
    [0.0734637681159421 0.797619047619048 0.0293030303030303 0.160714285714286],...
    'String',{'a'},...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.301395256916996 0.803571428571428 0.0293030303030302 0.160714285714286],...
    'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.535914361001318 0.803571428571428 0.0293030303030302 0.160714285714286],...
    'String','c',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.763845849802372 0.809523809523809 0.0293030303030303 0.160714285714286],...
    'String','d',...
    'FitBoxToText','off',...
    'EdgeColor','none');

%% Plot 3: EIS trends vs. temperature, SOC
% ax1: BOL vs T
Data_ax1 = Data2(Data2.seriesIdx == 44, :);
Data_ax1 = sortrows(Data_ax1, 'TdegC_EIS');
% ax2: Aged vs T
Data_ax2 = Data(Data.seriesIdx == 16, :);
Data_ax2 = Data_ax2(Data_ax2.N == 1650,:);
% ax3: BOL vs soc
Data_ax3 = Data2(Data2.seriesIdx == 43, :);
Data_ax3 = Data_ax3(Data_ax3.TdegC_EIS == 0, :);
% ax4: Aged vs soc
Data_ax4 = Data2(Data2.seriesIdx == 40, :); %37,38
Data_ax4 = Data_ax4(Data_ax4.TdegC_EIS == 0, :);

% colors
colorsT = viridis(height(Data_ax1));
temps = Data_ax1.TdegC_EIS;
colorsSOC = cividis(height(Data_ax3));

% plots
figure; t = tiledlayout('flow', 'Padding', 'compact', 'TileSpacing', 'compact');
% Plot BOL vs T
nexttile; D = Data_ax1; hold on; box on; grid on;
for i = 1:height(D)
    c = colorsT(i,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0 0.008].*1e4); ylim([-0.005 0.003].*1e4); axis('square')
% Plot Aged vs T
nexttile; D = Data_ax2; hold on; box on; grid on;
for i = 1:height(D)
    idxColor = temps == D.TdegC_EIS(i);
    c = colorsT(idxColor,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0 0.008].*1e4); ylim([-0.005 0.003].*1e4); axis('square')


tickLabels = temps;
ticklabels = compose("%d", tickLabels);
c = colorbar('Colormap', colorsT, 'Limits', [0 1],...
    'Ticks', linspace(0,1,length(temps)), 'TickLabels', ticklabels);
c.Label.String = 'Temperature (\circC)';
c.Label.Position = [-1.065833330154419,0.494624137878418,0];
c.TickLength = 0.02;
c.Layout.Tile = 'east';

set(gcf, 'Units', 'inches', 'Position', [4.697916666666666,6.65625,4.3,1.75]);

annotation(gcf,'textbox',...
    [0.557900726392252 0.803571428571428 0.0667966101694913 0.154761904761905],...
    'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.13417191283293 0.80952380952381 0.0667966101694915 0.154761904761905],...
    'String',{'a'},...
    'FitBoxToText','off',...
    'EdgeColor','none');

figure; t = tiledlayout('flow', 'Padding', 'compact', 'TileSpacing', 'compact');
% Plot BOL vs T
nexttile; D = Data_ax3; hold on; box on; grid on;
for i = 1:height(D)
    c = colorsSOC(i,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0 0.007].*1e4); ylim([-0.004 0.003].*1e4); axis('square')
% Plot Aged vs T
nexttile; D = Data_ax4; hold on; box on; grid on;
for i = 1:height(D)
    c = colorsSOC(i,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0 0.009].*1e4); ylim([-0.005 0.004].*1e4); axis('square')


tickLabels = Data_ax4.soc_EIS.*100;
ticklabels = compose("%d%%", tickLabels);
c = colorbar('Colormap', colorsSOC, 'Limits', [0 1],...
    'Ticks', linspace(0.1,0.9,length(Data_ax4.soc_EIS)), 'TickLabels', ticklabels);
c.Label.String = 'State of charge';
c.Label.Position = [-1.065833330154419,0.494624137878418,0];
c.TickLength = 0.02;
c.Layout.Tile = 'east';

set(gcf, 'Units', 'inches', 'Position', [4.697916666666666,6.65625,4.3,1.75]);

annotation(gcf,'textbox',...
    [0.560322033898306 0.761904761904761 0.0667966101694913 0.154761904761905],...
    'String','d',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.14143583535109 0.761904761904761 0.0667966101694914 0.154761904761905],...
    'String','c',...
    'FitBoxToText','off',...
    'EdgeColor','none');

%% Plot 4: correlations
% 25C data = DataA1 (train), DataA2 (test)
%-10C Data
DataB = Data(Data.TdegC_EIS == -10, :);
% Training/CV data
maskTestB = any(DataB.seriesIdx == cellsTest,2);
DataB1 = DataB(~maskTestB, :);
% Test data
DataB2 = DataB(maskTestB, :);

figure; t = tiledlayout('flow', 'Padding', 'compact', 'TileSpacing', 'compact');
% ZReal -10C
nexttile; hold on; box on; grid on;
plot(freq, abs(corr(DataB1.q, DataB1.Zreal)), '-k')
plot(freq, abs(corr(DataB2.q, DataB2.Zreal)), ':k', 'LineWidth', 1.5)
xlabel(t,'Frequency (Hz)', 'FontSize', 10); ylabel(t,["Absolute correlation with";"rel. discharge capacity"], 'FontSize', 10);
title('Z_{Real} @ -10\circC (\Omega)'); lgd = legend('Training Data', 'Testing Data', 'NumColumns', 2); 
lgd.Layout.Tile = 'north'; ylim([0 1]); set(gca, 'XScale', 'log')
% Zimag -10C
nexttile; hold on; box on; grid on;
plot(freq, abs(corr(DataB1.q, DataB1.Zimag)), '-k')
plot(freq, abs(corr(DataB2.q, DataB2.Zimag)), ':k', 'LineWidth', 1.5)
title('Z_{Imaginary} @ -10\circC (\Omega)'); ylim([0 1]); set(gca, 'XScale', 'log')
% Zmag -10C
nexttile; hold on; box on; grid on;
plot(freq, abs(corr(DataB1.q, DataB1.Zmag)), '-k')
plot(freq, abs(corr(DataB2.q, DataB2.Zmag)), ':k', 'LineWidth', 1.5)
title('|Z| @ -10\circC (\Omega)'); ylim([0 1]); set(gca, 'XScale', 'log')
% Zphz -10C
nexttile; hold on; box on; grid on;
plot(freq, abs(corr(DataB1.q, DataB1.Zphz)), '-k')
plot(freq, abs(corr(DataB2.q, DataB2.Zphz)), ':k', 'LineWidth', 1.5)
title('\angleZ @ -10\circC (\circ)'); ylim([0 1]); set(gca, 'XScale', 'log')

% ZReal 25C
nexttile; hold on; box on; grid on;
plot(freq, abs(corr(DataA1.q, DataA1.Zreal)), '-k')
plot(freq, abs(corr(DataA2.q, DataA2.Zreal)), ':k', 'LineWidth', 1.5)
title('Z_{Real} @ 25\circC (\Omega)'); ylim([0 1]); set(gca, 'XScale', 'log')
% Zimag 25C
nexttile; hold on; box on; grid on;
plot(freq, abs(corr(DataA1.q, DataA1.Zimag)), '-k')
plot(freq, abs(corr(DataA2.q, DataA2.Zimag)), ':k', 'LineWidth', 1.5)
title('Z_{Imaginary} @ 25\circC (\Omega)'); ylim([0 1]); set(gca, 'XScale', 'log')
% Zmag 25C
nexttile; hold on; box on; grid on;
plot(freq, abs(corr(DataA1.q, DataA1.Zmag)), '-k')
plot(freq, abs(corr(DataA2.q, DataA2.Zmag)), ':k', 'LineWidth', 1.5)
title('|Z| @ 25\circC (\Omega)'); ylim([0 1]); set(gca, 'XScale', 'log')
% Zphz 25C
nexttile; hold on; box on; grid on;
plot(freq, abs(corr(DataA1.q, DataA1.Zphz)), '-k')
plot(freq, abs(corr(DataA2.q, DataA2.Zphz)), ':k', 'LineWidth', 1.5)
title('\angleZ @ 25\circC (\circ)'); ylim([0 1]); set(gca, 'XScale', 'log')

set(gcf, 'Units', 'inches', 'Position', [7.104166666666666,6.0625,6.5,3.25])

% one more time, but with a little less info. Earlier one might be too
% much.
figure; t = tiledlayout('flow', 'Padding', 'compact', 'TileSpacing', 'compact');
% ZReal
nexttile; hold on; box on; grid on;
%-10C
plot(freq, abs(corr(DataB1.q, DataB1.Zreal)), '-', 'Color', colortriplet(1,:), 'LineWidth', 2)
plot(freq, abs(corr(DataB2.q, DataB2.Zreal)), ':', 'Color', colortriplet(1,:), 'LineWidth', 2, 'AlignVertexCenters', 'on')
%25C
plot(freq, abs(corr(DataA1.q, DataA1.Zreal)), '-', 'Color', colortriplet(2,:), 'LineWidth', 2)
plot(freq, abs(corr(DataA2.q, DataA2.Zreal)), ':', 'Color', colortriplet(2,:), 'LineWidth', 2, 'AlignVertexCenters', 'on')
%decorations
xlabel('Frequency (Hz)', 'FontSize', 10); ylabel('|corr(q,Z_{Real}(f))|' , 'FontSize', 10);
% title('Z_{Real} (\Omega)'); 
lgd = legend('Train data (-10 \circC)', 'Test data (-10 \circC)', 'Train data (25 \circC)', 'Test data (25 \circC)'); 
lgd.Layout.Tile = 'east'; ylim([0 1]); set(gca, 'XScale', 'log')
% Zimag
nexttile; hold on; box on; grid on;
%-10C
plot(freq, abs(corr(DataB1.q, DataB1.Zimag)), '-', 'Color', colortriplet(1,:), 'LineWidth', 2)
plot(freq, abs(corr(DataB2.q, DataB2.Zimag)), ':', 'Color', colortriplet(1,:), 'LineWidth', 2, 'AlignVertexCenters', 'on')
%25C
plot(freq, abs(corr(DataA1.q, DataA1.Zimag)), '-', 'Color', colortriplet(2,:), 'LineWidth', 2)
plot(freq, abs(corr(DataA2.q, DataA2.Zimag)), ':', 'Color', colortriplet(2,:), 'LineWidth', 2, 'AlignVertexCenters', 'on')
%label
xlabel('Frequency (Hz)', 'FontSize', 10); ylabel('|corr(q,Z_{Imaginary}(f))|', 'FontSize', 10);
%title
% title('Z_{Imaginary} (\Omega)'); 
ylim([0 1]); set(gca, 'XScale', 'log')
set(gcf, 'Units', 'inches', 'Position', [7.104166666666666,6.0625,6.5,2])

%% Plot 5: examine extracted statistical features

% Statistical features:
load('data\Data_Denso2021.mat', 'DataFormatted')
X = DataFormatted(idxKeep, 26:end);
X = generateFeaturesStatistical(X);
w = size(X,2)/4;
stats = ["Variance", "Mean", "Median", "IQR", "MAD", "MdAD", "Range"];
stats = categorical(stats);

figure; t = tiledlayout(2, 4, 'Padding', 'compact', 'TileSpacing', 'compact');
% -10C
X_ = X(Data.TdegC_EIS == -10, :);
X1 = X_(:,1:w);
X2 = X_(:,w+1:(w*2));
X3 = X_(:,(w*2+1):(w*3));
X4 = X_(:,(w*3+1):end);
% ZReal -10C
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(DataB1.q, X1{~maskTestB,:})), 'ok', 'MarkerFaceColor', 'k')
plot(stats, abs(corr(DataB2.q, X1{maskTestB,:})), 'dr', 'LineWidth', 1.5)
ylabel(t,["Absolute correlation with";"rel. discharge capacity"], 'FontSize', 10);
title('Z_{Real} @ -10\circC (\Omega)'); lgd = legend('Training Data', 'Testing Data', 'NumColumns', 2); 
lgd.Layout.Tile = 'north'; ylim([0 1]);
% Zimag -10C
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(DataB1.q, X2{~maskTestB,:})), 'ok', 'MarkerFaceColor', 'k')
plot(stats, abs(corr(DataB2.q, X2{maskTestB,:})), 'dr', 'LineWidth', 1.5)
title('Z_{Imaginary} @ -10\circC (\Omega)'); ylim([0 1]);
% Zmag -10C
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(DataB1.q, X3{~maskTestB,:})), 'ok', 'MarkerFaceColor', 'k')
plot(stats, abs(corr(DataB2.q, X3{maskTestB,:})), 'dr', 'LineWidth', 1.5)
title('|Z| @ -10\circC (\Omega)'); ylim([0 1]);
% Zphz -10C
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(DataB1.q, X4{~maskTestB,:})), 'ok', 'MarkerFaceColor', 'k')
plot(stats, abs(corr(DataB2.q, X4{maskTestB,:})), 'dr', 'LineWidth', 1.5)
title('\angleZ @ -10\circC (\circ)'); ylim([0 1]);

% 25C
X_ = X(Data.TdegC_EIS == 25, :);
X1 = X_(:,1:w);
X2 = X_(:,w+1:(w*2));
X3 = X_(:,(w*2+1):(w*3));
X4 = X_(:,(w*3+1):end);
% ZReal 25C
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(DataA1.q, X1{~maskTestA,:})), 'ok', 'MarkerFaceColor', 'k')
plot(stats, abs(corr(DataA2.q, X1{maskTestA,:})), 'dr', 'LineWidth', 1.5)
title('Z_{Real} @ 25\circC (\Omega)'); ylim([0 1]);
% Zimag 25C
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(DataA1.q, X2{~maskTestA,:})), 'ok', 'MarkerFaceColor', 'k')
plot(stats, abs(corr(DataA2.q, X2{maskTestA,:})), 'dr', 'LineWidth', 1.5)
title('Z_{Imaginary} @ 25\circC (\Omega)'); ylim([0 1]);
% Zmag 25C
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(DataA1.q, X3{~maskTestA,:})), 'ok', 'MarkerFaceColor', 'k')
plot(stats, abs(corr(DataA2.q, X3{maskTestA,:})), 'dr', 'LineWidth', 1.5)
title('|Z| @ 25\circC (\Omega)'); ylim([0 1]);
% Zphz 25C
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(DataA1.q, X4{~maskTestA,:})), 'ok', 'MarkerFaceColor', 'k')
plot(stats, abs(corr(DataA2.q, X4{maskTestA,:})), 'dr', 'LineWidth', 1.5)
title('\angleZ @ 25\circC (\circ)'); ylim([0 1]);


set(gcf, 'Units', 'inches', 'Position', [7.104166666666666,6.0625,6.5,4])


% a different look:
figure; t = tiledlayout(1, 4, 'Padding', 'compact', 'TileSpacing', 'compact');
% -10C
X_ = X(Data.TdegC_EIS == -10, :);
X1 = X_(:,1:w);
X2 = X_(:,w+1:(w*2));
X3 = X_(:,(w*2+1):(w*3));
X4 = X_(:,(w*3+1):end);
% 25C
XX_ = X(Data.TdegC_EIS == 25, :);
XX1 = XX_(:,1:w);
XX2 = XX_(:,w+1:(w*2));
XX3 = XX_(:,(w*2+1):(w*3));
XX4 = XX_(:,(w*3+1):end);
% ZReal
nexttile; hold on; box on; grid on;
% -10C
plot(stats, abs(corr(DataB1.q, X1{~maskTestB,:})), 'o', 'LineWidth', 1.5, 'Color', colortriplet(1,:))
plot(stats, abs(corr(DataB2.q, X1{maskTestB,:})), 'd', 'LineWidth', 1.5, 'Color', colortriplet(1,:))
% 25C
plot(stats, abs(corr(DataA1.q, XX1{~maskTestA,:})), 'o', 'LineWidth', 1.5, 'Color', colortriplet(2,:))
plot(stats, abs(corr(DataA2.q, XX1{maskTestA,:})), 'd', 'LineWidth', 1.5, 'Color', colortriplet(2,:))
% decorations
ylabel(t,["Absolute correlation with";"C/3 rel. discharge capacity"], 'FontSize', 10);
title('Z_{Real}'); 
lgd = legend('Train data (-10 \circC)', 'Test data (-10 \circC)', 'Train data (25 \circC)', 'Test data (25 \circC)'); 
lgd.Layout.Tile = 'east'; 
ylim([0 1]);
% Zimag
nexttile; hold on; box on; grid on;
% -10C
plot(stats, abs(corr(DataB1.q, X2{~maskTestB,:})), 'o', 'LineWidth', 1.5, 'Color', colortriplet(1,:))
plot(stats, abs(corr(DataB2.q, X2{maskTestB,:})), 'd', 'LineWidth', 1.5, 'Color', colortriplet(1,:))
% 25C
plot(stats, abs(corr(DataA1.q, XX2{~maskTestA,:})), 'o', 'LineWidth', 1.5, 'Color', colortriplet(2,:))
plot(stats, abs(corr(DataA2.q, XX2{maskTestA,:})), 'd', 'LineWidth', 1.5, 'Color', colortriplet(2,:))
title('Z_{Imaginary}'); ylim([0 1]);
% Zmag
nexttile; hold on; box on; grid on;
% -10C
plot(stats, abs(corr(DataB1.q, X3{~maskTestB,:})), 'o', 'LineWidth', 1.5, 'Color', colortriplet(1,:))
plot(stats, abs(corr(DataB2.q, X3{maskTestB,:})), 'd', 'LineWidth', 1.5, 'Color', colortriplet(1,:))
% 25C
plot(stats, abs(corr(DataA1.q, XX3{~maskTestA,:})), 'o', 'LineWidth', 1.5, 'Color', colortriplet(2,:))
plot(stats, abs(corr(DataA2.q, XX3{maskTestA,:})), 'd', 'LineWidth', 1.5, 'Color', colortriplet(2,:))
title('|Z|'); ylim([0 1]);
% Zphz
nexttile; hold on; box on; grid on;
% -10C
plot(stats, abs(corr(DataB1.q, X4{~maskTestB,:})), 'o', 'LineWidth', 1.5, 'Color', colortriplet(1,:))
plot(stats, abs(corr(DataB2.q, X4{maskTestB,:})), 'd', 'LineWidth', 1.5, 'Color', colortriplet(1,:))
% 25C
plot(stats, abs(corr(DataA1.q, XX4{~maskTestA,:})), 'o', 'LineWidth', 1.5, 'Color', colortriplet(2,:))
plot(stats, abs(corr(DataA2.q, XX4{maskTestA,:})), 'd', 'LineWidth', 1.5, 'Color', colortriplet(2,:))
% title
title('\angleZ'); ylim([0 1]);
% size
set(gcf, 'Units', 'inches', 'Position', [7.104166666666666,6.0625,6.5,2])


%% Plot 6: examine extracted PCA, UMAP features
addpath('C:\Users\pgasper\Documents\GitHub\ml_utils_matlab')

%{
%UMAP %default = min_dist = 0.1, n_neighbors = 15
% More neighbors seems to get broader trends, mapping more nicely to
% capacity. Smaller min-dist also helps make clear groups, since there can
% be many measurements with similar EIS data
[reduction, umap, clusterIdentifiers, extras]=run_umap(zscore(X{:,1:138}),'min_dist',0.1,'n_neighbors',30,'verbose','none');
% better
[reduction, umap, clusterIdentifiers, extras]=run_umap(zscore(X{:,1:138}),'min_dist',0.05,'n_neighbors',30,'verbose','none');

% plotting 2D reduction
figure; gscatter(reduction(:,1), reduction(:,2), Data.seriesIdx, colors);
figure; scatter(reduction(:,1), reduction(:,2), 8, Data.q); colorbar();


% COMPARE TO PCA
%}

%umap params
min_dist = 0.3;
n_neighbors = 20;
% PCA on -10C
X = DataFormatted(idxKeep, [1,26:end]);
X = X(Data.TdegC_EIS == -10, :);
maskTest = any(X.seriesIdx == cellsTest, 2);
X = X(:, 2:139); %redundant Zreal/Zimag and Zmag/Zphz data
Xtrain = X(~maskTest, :); Xtest = X(maskTest,:);
[Xtrain, mu, sigma] = zscore(Xtrain{:,:});
% Transform
[coeffBtrain,scoreBtrain,~,~,explained,means] = pca(Xtrain);
umap = UMAP('min_dist',min_dist,'n_neighbors',n_neighbors);
umap = umap.fit(Xtrain);
reductionBtrain = umap.embedding;
% figure; plot(cumsum(explained(1:10))) % first ten features explain ~99.5% of the variance
Xtest = (Xtest{:,:}-mu)./sigma;
scoreBtest = (Xtest - means)*coeffBtrain(:,1:10);
reductionBtest = umap.transform(Xtest);

% PCA on 25C
X = DataFormatted(idxKeep, [1,26:end]);
X = X(Data.TdegC_EIS == 25, :);
maskTest = any(X.seriesIdx == cellsTest, 2);
X = X(:, 2:139); %redundant Zreal/Zimag and Zmag/Zphz data
Xtrain = X(~maskTest, :); Xtest = X(maskTest,:);
[Xtrain, mu, sigma] = zscore(Xtrain{:,:});
[coeffAtrain,scoreAtrain,~,~,explained,means] = pca(Xtrain);
umap = UMAP('min_dist',min_dist,'n_neighbors',n_neighbors);
umap = umap.fit(Xtrain);
reductionAtrain = umap.embedding;
% figure; plot(cumsum(explained(1:10))) % first ten features explain ~99.3% of the variance
Xtest = (Xtest{:,:}-mu)./sigma;
scoreAtest = (Xtest - means)*coeffAtrain(:,1:10);
reductionAtest = umap.transform(Xtest);

%{
% correlation of PCA scores
figure; t = tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
% -10C PCA
nexttile; hold on; box on; grid on;
plot(abs(corr(DataB1.q, scoreBtrain(:,1:10))), '-ok', 'MarkerFaceColor', 'k')
plot(abs(corr(DataB2.q, scoreBtest)), ':dr', 'LineWidth', 1.5)
ylabel(t,["Absolute correlation with rel. discharge capacity"], 'FontSize', 10);
xlabel(t,'Principal component index', 'FontSize', 10)
title('-10\circC EIS'); lgd = legend('Training Data', 'Testing Data', 'NumColumns', 2); 
lgd.Layout.Tile = 'north'; ylim([0 1]);
% 25C PCA
nexttile; hold on; box on; grid on;
plot(abs(corr(DataA1.q, scoreAtrain(:,1:10))), '-ok', 'MarkerFaceColor', 'k')
plot(abs(corr(DataA2.q, scoreAtest)), ':dr', 'LineWidth', 1.5)
title('25\circC EIS'); ylim([0 1]);
set(gcf, 'Units', 'inches', 'Position', [5,3,3.25,5])

% scatter plots of scores vs capacity
figure; t = tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%-10C 
y1 = DataB1.q; y2 = DataB2.q;
x1 = scoreBtrain;
x2 = scoreBtest;
% PCA 1
nexttile; hold on; box on; grid on;
plot(x1(:,1), y1, '.k', 'LineWidth', 1)
plot(x2(:,1), y2, '+r', 'LineWidth', 1)
title('-10\circC component 1')
xlabel(t, 'Component score', 'FontSize', 10)
ylabel(t, 'C/3 relative discharge capacity', 'FontSize', 10)
lgd = legend('Training Data', 'Testing Data', 'NumColumns', 2); 
lgd.Layout.Tile = 'north';
% PCA 2
nexttile; hold on; box on; grid on;
plot(x1(:,2), y1, '.k', 'LineWidth', 1)
plot(x2(:,2), y2, '+r', 'LineWidth', 1)
title('-10\circC component 2')

%25C 
y1 = DataA1.q; y2 = DataA2.q;
x1 = scoreAtrain;
x2 = scoreAtest;
% PCA 1
nexttile; hold on; box on; grid on;
plot(x1(:,1), y1, '.k', 'LineWidth', 1)
plot(x2(:,1), y2, '+r', 'LineWidth', 1)
title('25\circC component 1')
xlim([-15 50])
% PCA 2
nexttile; hold on; box on; grid on;
plot(x1(:,2), y1, '.k', 'LineWidth', 1)
plot(x2(:,2), y2, '+r', 'LineWidth', 1)
title('25\circC component 2')
xlim([-30 30])

set(gcf, 'Units', 'inches', 'Position', [5,3,3.25,4])
%}


lw = 1.5;
colors = plasma(256);
% scatter plots of scores colored by series and capacity
figure; t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%-10C 
y1 = flipud(DataB1.q); y2 = flipud(DataB2.q);
x1 = flipud(scoreBtrain);
x2 = flipud(scoreBtest);
% % PCA (colored by series)
nexttile; hold on; box on; grid on;
% gscatter(x1(:,1), x1(:,2), DataB1.seriesIdx, colorsTrain, [], 10, 'off');
% gscatter(x2(:,1), x2(:,2), DataB2.seriesIdx, colorsTest, '+', 10, 'off');
xlabel(t, 'Component 1', 'FontSize', 10)
ylabel(t, 'Component 2', 'FontSize', 10)
% title('-10\circC PCA (cell series)')
% PCA (colored by capacity)
% nexttile; hold on; box on; grid on;
scatter(x1(:,1), x1(:,2), 25, y1, 's', 'filled');
scatter(x2(:,1), x2(:,2), 40, y2, 'x', 'LineWidth', lw);
colormap(colors);
title('-10\circC PCA')
lgd = legend('Training Data', 'Testing Data', 'NumColumns', 2); 
lgd.Layout.Tile = 'north';

%25C
y1 = flipud(DataA1.q); y2 = flipud(DataA2.q);
x1 = flipud(scoreAtrain);
x2 = flipud(scoreAtest);
% % PCA (colored by series)
% nexttile; hold on; box on; grid on;
% gscatter(x1(:,1), x1(:,2), DataA1.seriesIdx, colorsTrain, [], 10, 'off');
% gscatter(x2(:,1), x2(:,2), DataA2.seriesIdx, colorsTest, '+', 10, 'off');
% axis([-15 50 -50 50])
% xlabel(t, 'Component 1', 'FontSize', 10)
% ylabel(t, 'Component 2', 'FontSize', 10)
% title('25\circC PCA (cell series)')
% % % % lgd = legend('Training Data', 'Testing Data', 'NumColumns', 2); 
% % % % lgd.Layout.Tile = 'north';
% PCA (colored by capacity)
nexttile; hold on; box on; grid on;
scatter(x1(:,1), x1(:,2), 25, y1, 's', 'filled');
scatter(x2(:,1), x2(:,2), 40, y2, 'x', 'LineWidth', lw);
axis([-15 50 -50 50]); yticks([-50 -30 -10 10 30 50]);
title('25\circC PCA')

cb = colorbar(); cb.Label.String = 'Rel. discharge capacity';
cb.Layout.Tile = 'east';
cb.Label.Position = [-1.073333263397217,0.826218433420054,0];

set(gcf, 'Units', 'inches', 'Position', [7.104166666666666,6.0625,6.5,3])


figure; t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
% same, but for UMAP
% scatter plots of scores colored by series and capacity
% % % figure; t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%-10C 
y1 = DataB1.q; y2 = DataB2.q;
x1 = reductionBtrain;
x2 = reductionBtest;
% % PCA (colored by series)
% nexttile; hold on; box on; grid on;
% gscatter(x1(:,1), x1(:,2), DataB1.seriesIdx, colorsTrain, [], 10, 'off');
% gscatter(x2(:,1), x2(:,2), DataB2.seriesIdx, colorsTest, '+', 10, 'off');
% xlabel(t, 'Component 1', 'FontSize', 10)
% ylabel(t, 'Component 2', 'FontSize', 10)
% title('-10\circC UMAP (cell series)')
% % % % lgd = legend('Training Data', 'Testing Data', 'NumColumns', 2); 
% % % % lgd.Layout.Tile = 'north';
% PCA (colored by capacity)
nexttile; hold on; box on; grid on;
scatter(x1(:,1), x1(:,2), 25, y1, 's', 'filled');
scatter(x2(:,1), x2(:,2), 40, y2, 'x', 'LineWidth', lw);
title('-10\circC UMAP')

colormap(colors);

%25C
y1 = DataA1.q; y2 = DataA2.q;
x1 = reductionAtrain;
x2 = reductionAtest;
% % PCA (colored by series)
% nexttile; hold on; box on; grid on;
% gscatter(x1(:,1), x1(:,2), DataA1.seriesIdx, colorsTrain, [], 10, 'off');
% gscatter(x2(:,1), x2(:,2), DataA2.seriesIdx, colorsTest, '+', 10, 'off');
% xlabel(t, 'Component 1', 'FontSize', 10)
% ylabel(t, 'Component 2', 'FontSize', 10)
% title('25\circC UMAP (cell series)')
% % % % lgd = legend('Training Data', 'Testing Data', 'NumColumns', 2); 
% % % % lgd.Layout.Tile = 'north';
% PCA (colored by capacity)
nexttile; hold on; box on; grid on;
scatter(x1(:,1), x1(:,2), 25, y1, 's', 'filled');
scatter(x2(:,1), x2(:,2), 40, y2, 'x', 'LineWidth', lw);
cb = colorbar(); cb.Label.String = 'Rel. discharge capacity';
cb.Label.Position = [-1.073333263397217,0.826218433420054,0];
title('25\circC UMAP')


%% Fig 7 - example of graphical features from -10C data
% Grab a single EIS measurement, show graphical features
D = Data(Data.seriesIdx == 16, :);
D = D(1,:);

%-10C
figure; t = tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile; box on; grid on; hold on;
plot(D.Zreal.*1e4, D.Zimag.*1e4, '-k', 'LineWidth', 2)
plot(D.Zreal(idxDecades).*1e4, D.Zimag(idxDecades).*1e4, 'dk', 'MarkerSize', 8, 'MarkerFaceColor', 'k')
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0 0.008].*1e4); ylim([-0.005 0.003].*1e4); axis('square')

% feature '0': Z values at f=1E4

% feature 1: min(Zreal) at f<1E4
Zreal_ = D.Zreal;
Zreal_(freq>1E4) = Inf;
[~, idx1] = min(Zreal_);
plot(D.Zreal(idx1).*1e4, D.Zimag(idx1).*1e4, 'or', 'LineWidth', 2)

% feature 2: max(-Zimag(f<10^2 & f>10^-1))
Zimag_ = D.Zimag;
Zimag_(freq>10^2 | freq<10^-1) = Inf;
[~, idx2] = max(-1.*Zimag_);
plot(D.Zreal(idx2).*1e4, D.Zimag(idx2).*1e4, 'or', 'LineWidth', 2)

% feature 3: min(-Zimag(f<10^0 & f>10^-2)
Zimag_ = D.Zimag;
Zimag_(freq>10^0 | freq<10^-2) = -Inf;
[~, idx3] = min(-1.*Zimag_);
plot(D.Zreal(idx3).*1e4, D.Zimag(idx3).*1e4, 'or', 'LineWidth', 2)

% feature 5: Z values at f(end)
plot(D.Zreal(end).*1e4, D.Zimag(end).*1e4, 'or', 'LineWidth', 2)

% 25C
% Grab a single EIS measurement, show graphical features
D = Data(Data.seriesIdx == 16, :);
D = D(26,:);

nexttile; box on; grid on; hold on;
plot(D.Zreal.*1e4, D.Zimag.*1e4, '-k', 'LineWidth', 2)
plot(D.Zreal(idxDecades).*1e4, D.Zimag(idxDecades).*1e4, 'dk', 'MarkerSize', 8, 'MarkerFaceColor', 'k')
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0.0006 0.0016].*1e4); ylim([-0.001 0.001].*1e4); axis('square')

% feature '0': Z values at f=1E4

% feature 1: min(Zreal) at f<1E4
Zreal_ = D.Zreal;
Zreal_(freq>1E4) = Inf;
[~, idx1] = min(Zreal_);
plot(D.Zreal(idx1).*1e4, D.Zimag(idx1).*1e4, 'or', 'LineWidth', 2)

% feature 2: Z at f=10^2 Hz
idx2 = freq == 100;
plot(D.Zreal(idx2).*1e4, D.Zimag(idx2).*1e4, 'or', 'LineWidth', 2)

% feature 3: Z at f=10^0 Hz
idx3 = freq == 1;
plot(D.Zreal(idx3).*1e4, D.Zimag(idx3).*1e4, 'or', 'LineWidth', 2)

% feature 5: Z values at f(end)
plot(D.Zreal(end).*1e4, D.Zimag(end).*1e4, 'or', 'LineWidth', 2)

set(gcf, 'Units', 'inches', 'Position', [5,3,3.25,5])

annotation(gcf,'textarrow',[0.676282051282051 0.666666666666667],...
    [0.691666666666667 0.73125],'String',{'-Z_{Imaginary}','valley'},...
    'HorizontalAlignment','center');
annotation(gcf,'textarrow',[0.448717948717949 0.442307692307692],...
    [0.664583333333333 0.614583333333334],'String',{'f = 10^4'},...
    'HorizontalAlignment','center');
annotation(gcf,'textarrow',[0.400641025641026 0.33974358974359],...
    [0.739583333333333 0.711111111111111],'String',{'Min Z_{Real}'});
annotation(gcf,'textarrow',[0.400641025641026 0.30448717948718],...
    [0.252083333333333 0.221527777777778],'String',{'Min Z_{Real}'});
annotation(gcf,'textarrow',[0.391025641025641 0.384615384615385],...
    [0.175 0.125],'String',{'f = 10^4'},'HorizontalAlignment','center');
annotation(gcf,'textarrow',[0.689102564102564 0.711538461538462],...
    [0.86875 0.816666666666667],'String',{'Lowest','frequency'},...
    'HorizontalAlignment','center');
annotation(gcf,'textarrow',[0.448717948717949 0.387820512820512],...
    [0.416666666666667 0.320833333333334],'String',{'f = 10^0'},...
    'HorizontalAlignment','center');
annotation(gcf,'textarrow',[0.333333333333333 0.330128205128205],...
    [0.3625 0.31875],'String',{'f = 10^2'},'HorizontalAlignment','center');
annotation(gcf,'textarrow',[0.682692307692308 0.682692307692308],...
    [0.347916666666667 0.4125],'String',{'Lowest','frequency'},...
    'HorizontalAlignment','center');
annotation(gcf,'textarrow',[0.461538461538462 0.493589743589744],...
    [0.875 0.827083333333333],'String',{'-Z_{Imaginary}','peak'},...
    'HorizontalAlignment','center');
annotation(gcf,'textbox',...
    [0.241384615384615 0.90625 0.0791282051282051 0.05625],'String',{'a'},...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.244589743589743 0.422916666666667 0.0791282051282051 0.05625],...
    'String','b',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');

%% UMAP, data at all temps and socs
DataAll = [Data(:,[17,24,25,27,28]); Data2(:,[3,4,5,7,8])];

X = DataAll{:,{'Zreal','Zimag'}}; X = zscore(X);

min_dist = 0.1;
n_neighbors = 30;
umap = UMAP('min_dist',min_dist,'n_neighbors',n_neighbors);
umap = umap.fit(X); 
reduction = umap.embedding;

% capacity colors
figure; hold on; box on; grid on;
colormap(plasma(256));
scatter(reduction(:,1), reduction(:,2), 25, DataAll.q, 's', 'filled');
cb = colorbar(); cb.Label.String = 'Rel. discharge capacity';
xlabel('Component 1'); ylabel('Component 2');
set(gcf, 'Units', 'inches', 'Position', [5.52083333333333,3.7187,3.25,2.5]);

% temperature colors
figure; hold on; box on; grid on;
colormap(viridis(256));
scatter(reduction(:,1), reduction(:,2), 25, DataAll.TdegC_EIS, 's', 'filled');
cb = colorbar(); cb.Label.String = 'Temperature (\circC)';
xlabel('Component 1'); ylabel('Component 2');
set(gcf, 'Units', 'inches', 'Position', [5.52083333333333,3.7187,3.25,2.5]);

% soc colors
figure; hold on; box on; grid on;
colormap(cividis(256));
scatter(reduction(:,1), reduction(:,2), 25, DataAll.soc_EIS, 's', 'filled');
cb = colorbar(); cb.Label.String = 'State of charge';
xlabel('Component 1'); ylabel('Component 2');
set(gcf, 'Units', 'inches', 'Position', [5.52083333333333,3.7187,3.25,2.5]);

%% DC resistance versus capacity
load('data\Denso\DensoData.mat', 'DensoData');

DensoData = DensoData(~isnan(DensoData.Rc25C50soc10s), :);
q = DensoData.q;
r25C = DensoData(:, [41:46]);
r25C.r0p01s = mean(r25C{:,[1,4]},2);
r25C.r0p1s = mean(r25C{:,[2,5]},2);
r25C.r10s = mean(r25C{:,[3,6]},2);
r25C = r25C(:, 7:end);
rm10C = DensoData(:, [53:58]);
rm10C.r0p01s = mean(rm10C{:,[1,4]},2);
rm10C.r0p1s = mean(rm10C{:,[2,5]},2);
rm10C.r10s = mean(rm10C{:,[3,6]},2);
rm10C = rm10C(:, 7:end);

% 25C
figure; t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(r25C.r0p01s, q, '.k')
xlabel('0.01s DC pulse resistance (\Omega)')
nexttile;
plot(r25C.r0p1s, q, '.k')
xlabel('0.1s DC pulse resistance (\Omega)')
nexttile;
plot(r25C.r10s, q, '.k')
xlabel('10s DC pulse resistance (\Omega)')
% decorations
ylabel(t, 'Rel. discharge capacity')
title(t, 'Discharge capacity vs. 25 \circC DC pulse resistance')
set(gcf, 'Units', 'inches', 'Position', [5.52083333333333,3.7187,6.5,2.5]);

% m10C
figure; t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(rm10C.r0p01s, q, '.k')
xlabel('0.01s DC pulse resistance (\Omega)')
nexttile;
plot(rm10C.r0p1s, q, '.k')
xlabel('0.1s DC pulse resistance (\Omega)')
nexttile;
plot(rm10C.r10s, q, '.k')
xlabel('10s DC pulse resistance (\Omega)')
% decorations
ylabel(t, 'Rel. discharge capacity')
title(t, 'Discharge capacity vs. -10 \circC DC pulse resistance')
set(gcf, 'Units', 'inches', 'Position', [5.52083333333333,3.7187,6.5,2.5]);

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

function plotCorrelations(DataTrain, DataTest, freq)
variableNames = DataTrain.Properties.VariableNames;
figure;
% ZReal
nexttile; hold on; box on; grid on;
plot(freq, abs(corr(DataTrain.Qb3, DataTrain{:, contains(variableNames, 'ZReal')})), '-k');
plot(freq, abs(corr(DataTest.Qb3, DataTest{:, contains(variableNames, 'ZReal')})), '--k');
xlabel('Frequency (Hz)'); ylabel(["Absolute correlation with";"discharge capacity"]);
title('Z_{Real}'); lgd = legend('Training Data', 'Testing Data', 'NumColumns', 2); 
lgd.Layout.Tile = 'north'; ylim([0 1]); set(gca, 'XScale', 'log')
% ZImag
nexttile; hold on; box on; grid on;
plot(freq, abs(corr(DataTrain.Qb3, DataTrain{:, contains(variableNames, 'ZImag')})), '-k');
plot(freq, abs(corr(DataTest.Qb3, DataTest{:, contains(variableNames, 'ZImag')})), '--k');
xlabel('Frequency (Hz)'); ylabel(["Absolute correlation with";"discharge capacity"]); 
title('Z_{Imag}'); ylim([0 1]); set(gca, 'XScale', 'log')
end