% Paul Gasper, NREL, 2/2022
% A place to make data exploration figures for the article.
clc; clear; close all;

% frequency vector
load('data\Data_Denso2021.mat', 'Data')
freq = Data.Freq(1,:); clearvars Data

% Load the data tables
load('data\Data_Denso2021_AllRPTs.mat', 'Data', 'Data2')
% Remove series 43 and 44 (cells measured at BOL and not aged)
Data(Data.seriesIdx == 43, :) = [];
Data(Data.seriesIdx == 44, :) = [];

% methods file path
addpath(genpath('functions'))

%% Plot 1: overview of the data
% a - q vs t
% b - r vs t
% c - q vs r
% d - q vs r, polyfit
% e - q vs dq/dr from polyfit
% f - q vs 0.01s r
% g - q vs 0.1s r
% h - q vs 10s r

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

colortriplet = [colors1(3,:); colors2(12,:); colors3(2,:)];

PlotOpt = setPlotOpt(...
    'DataSeriesLabelVar', 'name',...
    'DataSeriesLabelFormat', 'none',...
    'Colorbar','off');
DataLineProp = setLineProp('-','Marker','.',...
    'LineWidth',1,'MarkerSize',12,...
    'Color', colors);
plotData(DataA, 't', 'q', DataLineProp, PlotOpt)
ylabel('C/3 relative capacity (25\circC)')
% Denote EIS measurements
maskEIS = ~logical(DataA.isInterpEIS);
plot(DataA.t(maskEIS), DataA.q(maskEIS), '+k', 'LineWidth', 2, 'MarkerSize', 8);
ylim([0.65 1.1]); xlim([0 500])
set(gcf, 'Units', 'inches', 'Position', [2,2,4,3]);
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.82151282051282 0.821969696969697 0.0791282051282054 0.0984848484848488],...
    'String','a',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName', 'Arial');
exportgraphics(gcf, 'figures/data_fig1a.eps', 'Resolution', 600)

% DC resistance
PlotOpt = setPlotOpt(...
    'DataSeriesLabelVar', 'name',...
    'DataSeriesLabelFormat', 'none',...
    'Colorbar','off');
DataLineProp = setLineProp('-','Marker','.',...
    'LineWidth',1,'MarkerSize',12,...
    'Color', colors);
plotData(DataA, 't', 'rm10C', DataLineProp, PlotOpt)
ylabel({'Rel. DC pulse resistance';'(-10\circC, 50% SOC)'})
% Denote EIS measurements
maskEIS = ~logical(DataA.isInterpEIS);
plot(DataA.t(maskEIS), DataA.rm10C(maskEIS), '+k', 'LineWidth', 2, 'MarkerSize', 8);
ylim([0.7 1.7]); xlim([0 500])
set(gcf, 'Units', 'inches', 'Position', [2,2,4,3]);
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.824599240265906 0.753787878787879 0.0791282051282054 0.0984848484848488],...
    'String','b',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName', 'Arial');
ax = gca; kids = ax.Children;
lgd = legend(kids([32,8,3,1]), {'Storage','Cycling','Drive cycle','EIS'}, 'Location', 'northwest');
title(lgd, 'Aging condition')
exportgraphics(gcf, 'figures/data_fig1b.eps', 'Resolution', 600)
%%
% c
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
set(gcf, 'Units', 'inches', 'Position', [2,2,3.25,3]);
set(gca, 'FontName', 'Arial')
% % ax = gca; kids = ax.Children;
% % lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Position', [0.525726495726496,0.588194444444444,0.342948717948718,0.234953703703704]);
% % title(lgd, 'Aging condition')
annotation(gcf,'textbox',...
    [0.827923076923076 0.830176767676767 0.0791282051282055 0.098484848484849],...
    'String','c',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
exportgraphics(gcf, 'figures/data_fig1c.eps', 'Resolution', 600)

% d
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
set(gcf, 'Units', 'inches', 'Position', [2,2,3.25,3]);
set(gca, 'FontName', 'Arial')
% ax = gca; kids = ax.Children;
% lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northeast');
% title(lgd, 'Aging condition')
annotation(gcf,'textbox',...
    [0.824717948717948 0.826704545454545 0.0791282051282055 0.098484848484849],...
    'String','d',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
exportgraphics(gcf, 'figures/data_fig1d.eps', 'Resolution', 600)

% e - dq/dr for each line as a function of q
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
set(gcf, 'Units', 'inches', 'Position', [2,2,3.25,3]);
set(gca, 'FontName', 'Arial')
ylim([-10 10]); xlim([0.65 1])
annotation(gcf,'textbox',...
    [0.21574358974359 0.78472222222222 0.486179487179487 0.118055555555554],...
    'String','\downarrow resist. = \downarrow cap.',...
    'FontSize',10,...
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
    'FontSize',10,...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.824717948717948 0.827777777777778 0.0791282051282055 0.098484848484849],...
    'String','e',...
    'FontSize',10,...
    'FontName','Arial',...
    'FitBoxToText','off',...
    'EdgeColor','none');
exportgraphics(gcf, 'figures/data_fig1e.eps', 'Resolution', 600)

% f-h: DCIR at different time constants
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
set(gcf, 'Units', 'inches', 'Position', [2,2,3.666,3]);
set(gca, 'FontName', 'Arial')
% ax = gca; kids = ax.Children;
% lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northeast');
% title(lgd, 'Aging condition')
annotation(gcf,'textbox',...
    [0.824717948717948 0.826704545454545 0.0791282051282055 0.098484848484849],...
    'String','f',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
exportgraphics(gcf, 'figures/data_fig1f.eps', 'Resolution', 600)

%0p1s
plotData(DensoData, 'rm10C50soc0p1s', 'q', DataLineProp, PlotOpt)
% ylabel({'Rel. C/3 discharge';'capacity (25\circC)'})
ylabel('')
xlabel({'Rel. DC pulse resistance';'(0.01-0.1 s, -10\circC, 50% SOC)'})
set(gcf, 'Units', 'inches', 'Position', [2,2,3.25,3]);
set(gca, 'FontName', 'Arial')
% ax = gca; kids = ax.Children;
% lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northeast');
% title(lgd, 'Aging condition')
annotation(gcf,'textbox',...
    [0.824717948717948 0.826704545454545 0.0791282051282055 0.098484848484849],...
    'String','g',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
exportgraphics(gcf, 'figures/data_fig1g.eps', 'Resolution', 600)

%10s
plotData(DensoData, 'rm10C50soc10s', 'q', DataLineProp, PlotOpt)
% ylabel({'Rel. C/3 discharge';'capacity (25\circC)'})
ylabel('')
xlabel({'Rel. DC pulse resistance';'(0.1-10 s, -10\circC, 50% SOC)'})
set(gcf, 'Units', 'inches', 'Position', [2,2,3.25,3]);
set(gca, 'FontName', 'Arial')
xticks([0,1,3,5,7]);
% ax = gca; kids = ax.Children;
% lgd = legend(kids([28,15,2]), {'Storage','Cycling','Drive cycle'}, 'Location', 'northeast');
% title(lgd, 'Aging condition')
annotation(gcf,'textbox',...
    [0.824717948717948 0.826704545454545 0.0791282051282055 0.098484848484849],...
    'String','h',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial');
exportgraphics(gcf, 'figures/data_fig1h.eps', 'Resolution', 600)

%% Plot 2: EIS trends
% For 1 storage, 1 cycling cell, plot EIS curves for -10C and 25C data
% Color by capacity (capacity map: 0.6:1) (colormap: cividis) (solid line w/ markers at frequency decades)
capacity = linspace(1, 0.6, 256);
colors = plasma(256);

% indices of frequency decades
idxDecades = log10(freq) == round(log10(freq));

% Storage cell: cell 1
DataEIS1 = Data(Data.seriesIdx == 1 & Data.isInterpEIS == 0, :);
DataEIS1_m10C = DataEIS1(DataEIS1.TdegC_EIS == -10, :);
DataEIS1_25C = DataEIS1(DataEIS1.TdegC_EIS == 25, :);

figure; t = tiledlayout(1,4, 'Padding', 'none', 'TileSpacing', 'none');
% Plot -10C nyquist
nexttile; D = DataEIS1_m10C; hold on; box on; grid on;
for i = 1:height(D)
    [~, idxColor] = min(abs(capacity - D.q(i)));
    % raw EIS
    c = colors(idxColor,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel(t,'Z_{Imaginary} (10^{-4}\Omega)', 'FontSize', 9.35);
set(gca, 'YDir', 'reverse');
set(gca, 'FontName', 'Arial')
xlim([0 0.008].*1e4); ylim([-0.005 0.003].*1e4); axis('square')
% Plot 25C nyquist
nexttile; D = DataEIS1_25C; hold on; box on; grid on;
for i = 1:height(D)
    [~, idxColor] = min(abs(capacity - D.q(i)));
    % raw EIS
    c = colors(idxColor,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); %ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
set(gca, 'FontName', 'Arial')
xlim([0.0006 0.0012].*1e4); ylim([-0.00045 0.00015].*1e4); axis('square')

% Cycling cell: cell 16
DataEIS2 = Data(Data.seriesIdx == 16 & Data.isInterpEIS == 0, :);
DataEIS2_m10C = DataEIS2(DataEIS2.TdegC_EIS == -10, :);
DataEIS2_25C = DataEIS2(DataEIS2.TdegC_EIS == 25, :);

% Plot -10C nyquist
nexttile; D = DataEIS2_m10C; hold on; box on; grid on;
for i = 1:height(D)
    [~, idxColor] = min(abs(capacity - D.q(i)));
    % raw EIS
    c = colors(idxColor,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); %ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
set(gca, 'FontName', 'Arial')
xlim([0 0.008].*1e4); ylim([-0.005 0.003].*1e4); axis('square')
% Plot 25C nyquist
nexttile; D = DataEIS2_25C; hold on; box on; grid on;
for i = 1:height(D)
    [~, idxColor] = min(abs(capacity - D.q(i)));
    % raw EIS
    c = colors(idxColor,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); %ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
set(gca, 'FontName', 'Arial')
xlim([0.0006 0.0012].*1e4); ylim([-0.00045 0.00015].*1e4); axis('square')

ticklabels = linspace(floor(min(capacity)*100)/100, 1, 5);
c = colorbar('Colormap', flipud(colors), 'Limits', [0 1],...
    'Ticks', [0 0.25 0.5 0.75 1], 'TickLabels', ticklabels);
c.Label.String = 'Rel. discharge capacity';
c.Label.FontName = 'Arial'; c.Label.FontSize = 9.35;
c.Label.Position = [-1.06,0.494624137878418,0];
c.TickLength = 0.02;
c.Layout.Tile = 'east';

annotation(gcf,'textbox',...
    [0.0734637681159421 0.797619047619048 0.0293030303030303 0.160714285714286],...
    'String',{'a'},...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);
annotation(gcf,'textbox',...
    [0.295 0.797619047619048 0.0293030303030302 0.160714285714286],...
    'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);
annotation(gcf,'textbox',...
    [0.515 0.797619047619048 0.0293030303030302 0.160714285714286],...
    'String','c',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);
annotation(gcf,'textbox',...
    [0.74 0.797619047619048 0.0293030303030303 0.160714285714286],...
    'String','d',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);

set(gcf, 'Units', 'inches', 'Position', [1,1,8,2]);

exportgraphics(gcf, 'figures/data_fig2abcd.eps', 'Resolution', 600)

%%
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
figure; t = tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'none');
% Plot BOL vs T
nexttile; D = Data_ax1; hold on; box on; grid on;
for i = 1:height(D)
    c = colorsT(i,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel(t, 'Z_{Imaginary} (10^{-4}\Omega)', 'FontSize', 9.35);
set(gca, 'YDir', 'reverse');
set(gca, 'FontName', 'Arial')
xlim([0 0.008].*1e4); ylim([-0.005 0.003].*1e4); axis('square')
% Plot Aged vs T
nexttile; D = Data_ax2; hold on; box on; grid on;
for i = 1:height(D)
    idxColor = temps == D.TdegC_EIS(i);
    c = colorsT(idxColor,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); %ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
set(gca, 'FontName', 'Arial')
xlim([0 0.008].*1e4); ylim([-0.005 0.003].*1e4); axis('square')


ticklabels = temps;
ticklabels = compose("%d", ticklabels);
c = colorbar('Colormap', colorsT, 'Limits', [0 1],...
    'Ticks', linspace(0,1,length(temps)), 'TickLabels', ticklabels);
c.Label.String = 'Temperature (\circC)';
c.Label.Position = [-1.065833330154419,0.494624137878418,0];
c.Label.FontName = 'Arial';
c.Label.FontSize = 9.35;
c.TickLength = 0.02;
c.Layout.Tile = 'east';

set(gcf, 'Units', 'inches', 'Position', [2,2,4,2]);

annotation(gcf,'textbox',...
    [0.14 0.75 0.0667966101694913 0.154761904761905],...
    'String','e',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);
annotation(gcf,'textbox',...
    [0.53 0.75 0.0667966101694915 0.154761904761905],...
    'String',{'f'},...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);

exportgraphics(gcf, 'figures/data_fig2ef.eps', 'Resolution', 600)
%%
figure; t = tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'none');
% Plot BOL vs T
nexttile; D = Data_ax3; hold on; box on; grid on;
for i = 1:height(D)
    c = colorsSOC(i,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
set(gca, 'FontName', 'Arial')
xlim([0 0.007].*1e4); ylim([-0.004 0.003].*1e4); axis('square')
% Plot Aged vs T
nexttile; D = Data_ax4; hold on; box on; grid on;
for i = 1:height(D)
    c = colorsSOC(i,:);
    plot(D.Zreal(i,:).*1e4, D.Zimag(i,:).*1e4, '-', 'Color', c, 'LineWidth', 2)
    plot(D.Zreal(i,idxDecades).*1e4, D.Zimag(i,idxDecades).*1e4, 'd', 'MarkerSize', 6, 'Color', c, 'MarkerFaceColor', c)
end
xlabel('Z_{Real} (10^{-4}\Omega)'); %ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
set(gca, 'FontName', 'Arial')
xlim([0 0.009].*1e4); ylim([-0.005 0.004].*1e4); axis('square')


ticklabels = Data_ax4.soc_EIS.*100;
ticklabels = compose("%d%%", ticklabels);
c = colorbar('Colormap', colorsSOC, 'Limits', [0 1],...
    'Ticks', linspace(0.1,0.9,length(Data_ax4.soc_EIS)), 'TickLabels', ticklabels);
c.Label.String = 'State of charge';
c.Label.FontName = 'Arial';
c.Label.FontSize = 9.35;
c.Label.Position = [-1.065833330154419,0.494624137878418,0];
c.TickLength = 0.02;
c.Layout.Tile = 'east';

set(gcf, 'Units', 'inches', 'Position', [2,2,4,1.8]);

annotation(gcf,'textbox',...
    [0.14 0.75 0.0667966101694913 0.154761904761905],...
    'String','g',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);
annotation(gcf,'textbox',...
    [0.53 0.75 0.0667966101694915 0.154761904761905],...
    'String',{'h'},...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);

exportgraphics(gcf, 'figures/data_fig2gh.eps', 'Resolution', 600)

%% UMAP, data at all temps and socs
DataAll = [Data(~logical(Data.isInterpEIS),[17,24,25,27,28]); Data2(:,[3,4,5,7,8])];

X = DataAll{:,{'Zreal','Zimag'}}; X = zscore(X);

min_dist = 1;
n_neighbors = 20;
umap = UMAP('min_dist',min_dist,'n_neighbors',n_neighbors);
umap = umap.fit(X); 
reduction = umap.embedding;

markersize = 8; 

% capacity colors
figure; hold on; box on; grid on;
colormap(plasma(256));
scatter(reduction(:,1), reduction(:,2), markersize, DataAll.q, 'o', 'filled');
cb = colorbar(); cb.Label.String = 'Rel. discharge capacity'; cb.Label.FontSize = 10;
xlabel('Component 1'); ylabel('Component 2');
set(gcf, 'Units', 'inches', 'Position', [2,2,3.25,2.5]);
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.15 0.825 0.0695128205128205 0.0972222222222221],...
    'String','a',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 12);
exportgraphics(gcf, 'figures/data_fig3a.eps', 'Resolution', 600)

% temperature colors
figure; hold on; box on; grid on;
colormap(viridis(256));
scatter(reduction(:,1), reduction(:,2), markersize, DataAll.TdegC_EIS, 'o', 'filled');
cb = colorbar(); cb.Label.String = 'Temperature (\circC)'; cb.Label.FontSize = 10;
xlabel('Component 1'); ylabel('Component 2');
set(gcf, 'Units', 'inches', 'Position', [2,2,3.25,2.5]);
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.15 0.825 0.0695128205128205 0.0972222222222221],...
    'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 12);
exportgraphics(gcf, 'figures/data_fig3b.eps', 'Resolution', 600)

% soc colors
figure; hold on; box on; grid on;
colormap(cividis(256));
scatter(reduction(:,1), reduction(:,2), markersize, DataAll.soc_EIS, 'o', 'filled');
cb = colorbar(); cb.Label.String = 'State of charge'; cb.Label.FontSize = 10;
xlabel('Component 1'); ylabel('Component 2');
set(gcf, 'Units', 'inches', 'Position', [2,2,3.25,2.5]);
set(gca, 'FontName', 'Arial')
annotation(gcf,'textbox',...
    [0.15 0.825 0.0695128205128205 0.0972222222222221],...
    'String','c',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 12);
exportgraphics(gcf, 'figures/data_fig3c.eps', 'Resolution', 600)

%% Plot 4: correlations
% get rid of interp EIS data
Data = Data(~logical(Data.isInterpEIS),:);
DataTrain = Data(~any(Data.seriesIdx == cellsTest, 2), :);
DataTest = Data(any(Data.seriesIdx == cellsTest, 2), :);
% 25C data = DataA1 (train), DataA2 (test)
%-10C Data
DataB = Data(Data.TdegC_EIS == -10, :);
% Training/CV data
maskTestB = any(DataB.seriesIdx == cellsTest,2);
DataB1 = DataB(~maskTestB, :);
% Test data
DataB2 = DataB(maskTestB, :);

lw = 1.25;
figure; t = tiledlayout('flow', 'Padding', 'compact', 'TileSpacing', 'compact');
% ZReal
nexttile; hold on; box on; grid on;
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 0.5);
% all data
p1 = plot(freq, corr(DataTrain.q, DataTrain.Zreal), '-k', 'LineWidth', lw);
p2 = plot(freq, corr(DataTest.q, DataTest.Zreal), '--k', 'LineWidth', lw, 'AlignVertexCenters', 'on');
%-10C
p3 = plot(freq, corr(DataB1.q, DataB1.Zreal), '-', 'Color', colortriplet(1,:), 'LineWidth', lw);
p4 = plot(freq, corr(DataB2.q, DataB2.Zreal), '--', 'Color', colortriplet(1,:), 'LineWidth', lw);
%25C
p5 = plot(freq, corr(DataA1.q, DataA1.Zreal), '-', 'Color', colortriplet(2,:), 'LineWidth', lw);
p6 = plot(freq, corr(DataA2.q, DataA2.Zreal), '--', 'Color', colortriplet(2,:), 'LineWidth', lw);
%decorations
xlabel('Frequency (Hz)', 'FontSize', 10); ylabel('corr(q,Z_{Real}(f))' , 'FontSize', 10);
lgd = legend([p1 p2 p3 p4 p5 p6],...
    {'Training ', 'Test',...
    'Training (-10 \circC)', 'Test (-10 \circC)',...
    'Training (25 \circC)', 'Test (25 \circC)'});
lgd.Layout.Tile = 'east'; ylim([-1 1]); set(gca, 'XScale', 'log')
set(gca, 'FontName', 'Arial')

% Zimag
nexttile; hold on; box on; grid on;
yline(0, '-', 'Color', [0 0 0], 'LineWidth', 0.5);
% all data
plot(freq, corr(DataTrain.q, DataTrain.Zimag), '-k', 'LineWidth', lw)
plot(freq, corr(DataTest.q, DataTest.Zimag), '--k', 'LineWidth', lw, 'AlignVertexCenters', 'on')
%-10C
plot(freq, corr(DataB1.q, DataB1.Zimag), '-', 'Color', colortriplet(1,:), 'LineWidth', lw)
plot(freq, corr(DataB2.q, DataB2.Zimag), '--', 'Color', colortriplet(1,:), 'LineWidth', lw)
%25C
plot(freq, corr(DataA1.q, DataA1.Zimag), '-', 'Color', colortriplet(2,:), 'LineWidth', lw)
plot(freq, corr(DataA2.q, DataA2.Zimag), '--', 'Color', colortriplet(2,:), 'LineWidth', lw)
%label
xlabel('Frequency (Hz)', 'FontSize', 10); ylabel('corr(q,Z_{Imaginary}(f))', 'FontSize', 10);
ylim([-1 1]); set(gca, 'XScale', 'log')
set(gcf, 'Units', 'inches', 'Position', [2,2,8,2.5])
set(gca, 'FontName', 'Arial')

annotation(gcf,'textbox',...
    [0.35 0.788194444444444 0.044940170940171 0.145833333333333],...
    'String',{'a'},...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 12);
annotation(gcf,'textbox',...
    [0.75 0.795138888888888 0.044940170940171 0.145833333333333],...
    'String','b',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 12);

exportgraphics(gcf, 'figures/data_fig4ab.eps', 'Resolution', 600)

%% Plot 5: examine extracted statistical features

% Statistical features:
load('data\Data_Denso2021.mat', 'DataFormatted')
X = DataFormatted(:, 25:end);
X = generateFeaturesStatistical(X);
w = size(X,2)/4;
% separate by zreal, zimag, zmag, zphz
X1 = X(:,1:w);
X2 = X(:,w+1:(w*2));
X3 = X(:,(w*2+1):(w*3));
X4 = X(:,(w*3+1):end);
% test mask
maskTest = any(Data.seriesIdx == cellsTest, 2);
% xlabels
stats = ["Variance", "Mean", "Median", "IQR", "MAD", "MdAD", "Range"];
stats = categorical(stats);

figure; t = tiledlayout(1, 4, 'Padding', 'compact', 'TileSpacing', 'compact');
% ZReal
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(Data.q(~maskTest), X1{~maskTest,:})), 'ok', 'LineWidth', 1.5)
plot(stats, abs(corr(Data.q(maskTest), X1{maskTest,:})), 'dk', 'LineWidth', 1.5)
% decorations
ylabel(t,["Absolute correlation with";"C/3 rel. discharge capacity"], 'FontSize', 10);
title('Z_{Real}', 'FontWeight', 'normal'); 
lgd = legend('Train data', 'Test data'); 
ylim([0 1]);
set(gca, 'FontName', 'Arial')

% Zimag
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(Data.q(~maskTest), X2{~maskTest,:})), 'ok', 'LineWidth', 1.5)
plot(stats, abs(corr(Data.q(maskTest), X2{maskTest,:})), 'dk', 'LineWidth', 1.5)
title('Z_{Imaginary}', 'FontWeight', 'normal'); ylim([0 1]);
set(gca, 'FontName', 'Arial')
% Zmag
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(Data.q(~maskTest), X3{~maskTest,:})), 'ok', 'LineWidth', 1.5)
plot(stats, abs(corr(Data.q(maskTest), X3{maskTest,:})), 'dk', 'LineWidth', 1.5)
title('|Z|', 'FontWeight', 'normal'); ylim([0 1]);  
set(gca, 'FontName', 'Arial')
% Zphz
nexttile; hold on; box on; grid on;
plot(stats, abs(corr(Data.q(~maskTest), X4{~maskTest,:})), 'ok', 'LineWidth', 1.5)
plot(stats, abs(corr(Data.q(maskTest), X4{maskTest,:})), 'dk', 'LineWidth', 1.5)
% title
title('\angleZ', 'FontWeight', 'normal'); ylim([0 1]);
set(gca, 'FontName', 'Arial')
% size
set(gcf, 'Units', 'inches', 'Position', [2,2,8,2])

annotation(gcf,'textbox',...
    [0.085 0.68 0.0293030303030303 0.160714285714286],...
    'String',{'c'},...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);
annotation(gcf,'textbox',...
    [0.315 0.68 0.0293030303030302 0.160714285714286],...
    'String','d',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);
annotation(gcf,'textbox',...
    [0.54 0.68 0.0293030303030302 0.160714285714286],...
    'String','e',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);
annotation(gcf,'textbox',...
    [0.77 0.68 0.0293030303030303 0.160714285714286],...
    'String','f',...
    'FitBoxToText','off',...
    'EdgeColor','none',...
    'FontName','Arial',...
    'FontSize', 11);

exportgraphics(gcf, 'figures/data_fig4cdef.eps', 'Resolution', 600)

%% graphical features from -10C data
% Grab a single EIS measurement, show graphical features
D = Data(Data.seriesIdx == 16, :);
D = D(1,:);

%-10C
figure; box on; grid on; hold on;
plot(D.Zreal.*1e4, D.Zimag.*1e4, '-k', 'LineWidth', 2)
plot(D.Zreal(idxDecades).*1e4, D.Zimag(idxDecades).*1e4, 'dk', 'MarkerSize', 8, 'MarkerFaceColor', 'k')
xlabel('Z_{Real} (10^{-4}\Omega)'); ylabel('Z_{Imaginary} (10^{-4}\Omega)');
set(gca, 'YDir', 'reverse');
xlim([0 0.008].*1e4); ylim([-0.005 0.003].*1e4); axis('square')

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

set(gcf, 'Units', 'inches', 'Position', [5,3,3.25,2.5])
set(gca, 'FontName', 'Arial')

annotation(gcf,'textarrow',[0.438034188034188 0.414529914529915],...
    [0.313888888888889 0.233333333333333],'String',{'f = 10^4'},'FontName', 'Arial');
annotation(gcf,'textarrow',[0.408119658119658 0.331196581196581],...
    [0.462888888888889 0.422222222222222],'String',{'Min Z_{Real}'},'FontName', 'Arial');
annotation(gcf,'textarrow',[0.344017094017094 0.333333333333333],...
    [0.713888888888889 0.583333333333333],'String',{'f = 10^2'},'FontName', 'Arial');
annotation(gcf,'textarrow',[0.568376068376068 0.613247863247863],...
    [0.683333333333333 0.602777777777778],'String',{'f = 10^0'},'FontName', 'Arial');
annotation(gcf,'textarrow',[0.645299145299145 0.694444444444444],...
    [0.811111111111111 0.647222222222222],'String',{'Lowest freq.'},'FontName', 'Arial');

exportgraphics(gcf, 'figures/data_fig5.eps', 'Resolution', 600)