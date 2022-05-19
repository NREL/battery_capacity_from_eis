function [uniqueColors, colorBarOpts] = colorDataSeriesByVar(Data, varStr, cmap)
%[uniqueColors, colorBarOpts] = COLORDATASERIESBYVAR(Data, varStr, cmap)
% Create a colormap that colors each data series in Data (denoted by unique
% values of Data.seriesIdx) by its value of 'varStr', one color for
% each unique value of 'varStr' in Data. This is useful for coloring
% variables by their temperature, for instance.
% Inputs:
%   Data (table): Data set of multiple data series, with each data series
%     denoted by a unique value of the variable seriesIdx.
%   varStr (char): Variable in Data to color lines by.
%   cmap (char): Specify a colormap. Supports MATLAB colormaps, as well
%     as cmocean and brewermap colormaps (search Mathworks file exchange).
% Output:
%   uniqueColors (double): RGB triplets for each data series.
%   colorBarOpts (struct): Specs for plotting a colorbar

% Get mean/unique value of varStr for each data series:
seriesIdx = unique(Data.seriesIdx, 'stable');
numSeries = length(seriesIdx);
var = Data.(varStr);
if isnumeric(var)
    seriesVars = zeros(numSeries, 1);
else
    seriesVars = cell(numSeries, 1);
end
for iSeries = 1:numSeries
    if isnumeric(var)
        seriesVars(iSeries) = mean(var(Data.seriesIdx == seriesIdx(iSeries)));
    else
        temp = var(Data.seriesIdx == seriesIdx(iSeries));
        seriesVars{iSeries} = temp{1};
    end
end

% Check if the colormap is reversed:
if strncmp(cmap, '-', 1)
    cmap = cmap(2:end);
    flagReverseCmap = 1;
else
    flagReverseCmap = 0;
end

% Get only the unique values of seriesVars (data series with the same value
% of varStr will have the same color)
if isnumeric(seriesVars)
    % unique sometimes duplicates numeric values due to floating point
    % tolerances. Only keep numeric values that are actually 'different'
    % from one another.
    [uniqueSeriesVars, ~, origSeriesIdx] = uniquetol(seriesVars);
else
    [uniqueSeriesVars, ~, origSeriesIdx] = unique(seriesVars,'stable');
end
% Decide if this is a 'discrete' variable (like a series idx, or cell name)
% or a continuous variable (temperature, SOC, ....) or a discrete colormap,
% in which case, we have a unique color / tick label for each discrete
% value of the variable.
cmapsDiscrete = {'Spectral', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'lines', 'colorcube', 'prism', 'flag'};
flagDiscrete = iscell(uniqueSeriesVars) || contains(varStr, 'Idx') || any(strcmp(cmap, cmapsDiscrete));

% Check if the colormap is diverging:
if any(strcmp(cmap, {'topo','balance','delta','curl','diff','tarn',...
        'BrBG','PRGn','PiYG','PuOr','RdBu','RdGy','RdYlBu','RdYlGn','Spectral'}))
    flagDiverging = 1;
else
    flagDiverging = 0;
end
% Get a colormap, one color per unique value:
switch cmap
    case {'parula','jet','hsv','hot','cool','spring','summer',...
            'autumn','winter','gray','bone','copper','pink','lines',...
            'colorcube','prism','flag','white','turbo'}
        cmap = str2func(cmap);
        if flagDiscrete
            uniqueColors = cmap(length(uniqueSeriesVars));
            if flagReverseCmap
                uniqueColors = flipud(uniqueColors);
            end
            colorBarOpts = getDiscreteColorBarOpts(uniqueColors, uniqueSeriesVars, varStr);
        else
            [uniqueColors, colorBarOpts] = getContinuousVarColors(uniqueSeriesVars, cmap, varStr, flagReverseCmap, flagDiverging);
        end
    case {'BrBG','PRGn','PiYG','PuOr','RdBu','RdGy','RdYlBu','RdYlGn',...
            'Spectral','Accent','Dark2','Paired','Pastel1','Pastel2',...
            'Set1','Set2','Set3','Blues','BuGn','BuPu','GnBu',...
            'Greens','Greys','OrRd','Oranges','PuBu','PuBuGn',...
            'PuRd','Purples','RdPu','Reds','YlGn','YlGnBu',...
            'YlOrBr','YlOrRd'}
        if flagDiscrete
            uniqueColors = brewermap(cmap, length(uniqueSeriesVars));
            if flagReverseCmap
                uniqueColors = flipud(uniqueColors);
            end
            colorBarOpts = getDiscreteColorBarOpts(uniqueColors, uniqueSeriesVars, varStr);
        else
            cmap = @() brewermap(cmap, 256);
            [uniqueColors, colorBarOpts] = getContinuousVarColors(uniqueSeriesVars, cmap, varStr, flagReverseCmap, flagDiverging);
        end
    case {'thermal','haline','solar','ice','oxy','deep','dense',...
            'algae','matter','turbid','speed','amp','tempo','rain',...
            'balance','delta','curl','diff','tarn','phase','topo'}
        if flagDiscrete
            uniqueColors = cmocean(cmap, length(uniqueSeriesVars));
            if flagReverseCmap
                uniqueColors = flipud(uniqueColors);
            end
            colorBarOpts = getDiscreteColorBarOpts(uniqueColors, uniqueSeriesVars, varStr);
        else
            cmap = @() cmocean(cmap, 256);
            [uniqueColors, colorBarOpts] = getContinuousVarColors(uniqueSeriesVars, cmap, varStr, flagReverseCmap, flagDiverging);
        end
    otherwise
        error("Cannot find specified colormap function, check PlotOpt.Colormap input.")
end
uniqueColors = uniqueColors(origSeriesIdx,:);
end

function [uniqueColors, colorBarOpts] = getContinuousVarColors(uniqueSeriesVars, cmap, varStr, flagReverseCmap, flagDiverging)
% For a continuous variable, gets colors spaced according to their value
% within the range of the variable.  If the colormap is a diverging
% colormap, center the colormap at 0 (if the range crosses 0). Also returns
% details for making the colorbar in colorBarOpts. Reverses the cmap if
% desired.
colors = cmap(); % all 256 colors
% Sometimes the above doesn't work with matlab default colormaps. Not sure
% why. Check the size:
if size(colors, 1) < 256
    colors = cmap(256);
end
if flagReverseCmap
    colors = flipud(colors);
end
% If its a diverging colormap, the min/max/range needs to be symmetric
% about 0:
if flagDiverging
    varMin = min(uniqueSeriesVars);
    varMax = max(uniqueSeriesVars);
    if varMin < 0 && varMax > 0 % the values span 0, center at 0
        maxDist = max(abs([varMin, varMax]));
        varMin = -maxDist;
        varMax = maxDist;
    end
    varRange = varMax-varMin;
else
    % Non-diverging color map
    varMin = min(uniqueSeriesVars);
    varMax = max(uniqueSeriesVars);
    varRange = varMax-varMin;
end
% Find the uniqueColors, based on their value, by scaling to 1:256 (int):
scaledVars = uniqueSeriesVars - varMin; % move min to 0
scaledVars = (scaledVars./varRange).*255; % scale to 0:255
scaledVars = round(scaledVars) + 1; % Round, scale to 1:256
uniqueColors = colors(scaledVars,:);
if varRange > 5
    % Round the varMin and varMax for making limits, ticks:
    varMin = round(varMin);
    varMax = round(varMax);
    varRange = varMax-varMin;
end
% Create ticks:
if length(uniqueSeriesVars) <= 5
    % just use the sorted values of the uniqueSeriesVars as ticks
    if varRange > 5
        ticks = sort(round(uniqueSeriesVars));
    else
        ticks = sort(uniqueSeriesVars);
    end
else
    % 5 evenly spaced tick marks
    ticks = [0, 0.25, 0.5, 0.75, 1];
    ticks = varMin + ticks.*varRange;
end
ticks = unique(ticks);
% Create tick labels:
tickLabels = cell(size(ticks));
for iTick = 1:length(ticks)
    tickLabels(iTick) = {num2str(ticks(iTick),2)};
end
% Store the color bar specs:
colorBarOpts.Colormap = colors;
colorBarOpts.Ticks = ticks;
colorBarOpts.TickLabels = tickLabels;
colorBarOpts.Limits = [varMin, varMax];
colorBarOpts.LabelString = varStrToAxisLabel(varStr);
end

function colorBarOpts = getDiscreteColorBarOpts(colormap, uniqueSeriesVars, varStr)
colorBarOpts.Colormap = colormap;
colorBarOpts.Ticks = ((((1:length(uniqueSeriesVars))-1).*2)+1)./(length(uniqueSeriesVars)*2); % puts ticks in the middle of each discrete color in the colorbar
if iscell(uniqueSeriesVars)
    tickLabels = uniqueSeriesVars;
else
    tickLabels = cell(size(uniqueSeriesVars));
    for j = 1:length(uniqueSeriesVars)
        tickLabels(j) = {num2str(uniqueSeriesVars(j),2)};
    end
end
colorBarOpts.TickLabels = tickLabels;
colorBarOpts.Limits = [0 1];
colorBarOpts.LabelString = varStrToAxisLabel(varStr);
end

