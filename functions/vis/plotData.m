function plotData(Data, xStr, yStr, DataLineProp, PlotOpt)
%PLOTDATA(Data, xStr, yStr, DataLineProp, PlotOpt)
% Plots the variable xStr versus yStr from the table Data. This plotting
% function assumes that the Data is composed of several independent data
% series, denoted by a unique value of Data.seriesIdx. Various options for
% plotting mutliple data series, creating labels/legends, and setting line
% styles are set by the PlotOpt struct. Use the method setPlotOpt to easily
% create the formatting struct. Use the method setLineProp to easily create
% a line property struct.
% Input:
%   Data (table): Data table that was used when calculating the
%     FitResult being plotted, can be used to automatically
%     color data series and their grab their labels.
%   xStr (char): Name of the x variable to plot against from
%     the data table.
%   yStr (char): Name of the y variable to plot from the data table
%     the data table.
%   DataLineProp (struct): Line properties. Create a line
%     properties struct easily using the setLineProp function.
%     Struct setting data line properties, but the name 'Color'
%     and 'MarkerFaceColor' can also be the phrase 'By varStr',
%     which colors the line / marker by its unique (or average)
%     value for any xStr, using the colormap specified in
%     PlotOpt.Colormap. Also takes in the field 'LineSpec', which
%     is simply input into the plot function. The property
%     'DisplayName' will be ignored, as this is set based on
%     DataSeriesLabelFormat options.
%     Examples:
%       'LineSpec': 'ok', '-', ':sr' (any LineSpec input
%       for plot)
%       'LineWidth': 1.5 (sets LineSpec)
%       'MarkerFaceColor': 'k', or 'By TdegK'
%   PlotOpt (struct): Struct setting plot formatting options,
%     colors, and styles
%       'SeparateAxes': 'Off', 'By data series', 'By data group'
%       'AxisType': 'plot' (default), 'semilogx', 'semilogy', 'loglog'
%       'AxisLimits': [xMin xMax yMin yMax]
%       'xScaleMultiplier': make the x-scale bigger to
%         accomodate text labels or legends
%       'Xlabel': sets the xlabel. Plot xlabel is normally
%         grabbed using the varStrToAxisLabel static method
%         with the input xStr.
%       'Ylabel': sets the ylabel. Plot ylabel is normally
%         grabbed using the varStrToAxisLabel static method
%         with the input yStr from obj.
%       'Title': title for the figure.
%       'Colormap': specify which colormap function to use to
%         generate colors, if any of the colors for the data
%         labels, data points, or fit lines are specified as
%         'By xStr'. Accepts any MATLAB colormap function name,
%         as well as colormaps from the 'cmocean' or
%         'brewermap' functions off of the Mathworks File
%         Exchange.
%       'DataSeriesLabelFormat': If 'IndividualDataSeriesAxes'
%         is 'On', the entries in Data.label are used to title
%         each data series plot. For 'Off' (single plot for all
%         data series), the entries in Data.label are plotted
%         following the rules described below for the following
%         options: 'label on plot', 'number on plot', 'in legend'
%           'off': No labels anywhere
%           'label on plot': Writes the text from each entry of
%             DataSeriesLabels next to the final point of each
%             data series. As labels are on the plot, a legend
%             is not shown for this case.
%           'number on plot': Writes the number of each data
%             series (its unique value of FitResult.seriesIdx)
%             next to the final point of each data series. A
%             legend is shown to the eastoutside of the plot
%             with the labels "number: Data.label{number}".
%           'in legend': No text on the plot denoting data
%             series. A legend is shown to the eastoutside of
%             the plot with the labels from Data.label.
%       'DataSeriesLabelVar': Table variable to grab labels from.
%       'DataSeriesLabelColor': Color for the data series
%         labels/numbers on the plot and in the legend. Can be
%         any valid MATLAB color ('k',[0 0 0]) or the phrase
%         'By varStr', which colors the label for each data series
%         according to its unique (or average) value of varStr,
%         using the colormap specified in PlotOpt.Colormap.
%         Default color is black.
%       'DataSeriesLabelFontSize': font size for all data labels
%         printed on the figure.

% Collect data:
if any(strcmp(Data.Properties.VariableNames, 'seriesIdx'))
    seriesIdx = Data.seriesIdx;
elseif any(strcmp(Data.Properties.VariableNames, 'groupIdx'))
    % getInvariantData by group removes the seriesIdx variable, just use
    % the groupIdx instead.
    Data.seriesIdx = Data.groupIdx;
    seriesIdx = Data.seriesIdx;
else
    error("Must have variables 'seriesIdx' or 'groupIdx' for plotting a data set.")
end
dataSeries = unique(seriesIdx, 'stable');
x = Data.(xStr);
y = Data.(yStr);

% Check if DataLineProp was input:
if nargin < 4 || isempty(DataLineProp)
    colors = lines(length(dataSeries));
    DataLineProp = setLineProp('-', 'Color', colors,...
        'Marker', '.', 'MarkerSize', 16, 'LineWidth', 1);
end
% Check if PlotOpt was input:
if nargin < 5 || isempty(PlotOpt)
    PlotOpt = setPlotOpt();
end

% Check Data line properties to get colors:
if isfield(PlotOpt, 'Colormap')
    [dataLineColors, dataMarkerColors, colorBarOpts] = getDataColors(Data, DataLineProp, PlotOpt.Colormap);
else
    [dataLineColors, dataMarkerColors, colorBarOpts] = getDataColors(Data, DataLineProp, []);
end

% Check some of the DataSeriesLabel properties, get colors if needed:
labelColors = getDataSeriesLabelColors(Data, PlotOpt);

% Set some flags based on plotting options:
[flagSeparateAxes, flagLabels, flagLabelsOnPlot] = setFlags(Data, PlotOpt);

% Create the figure:
figure
t = tiledlayout('flow','TileSpacing','compact');

% Plotting separate groups:
if flagSeparateAxes == 1
    % Identify each of the groups:
    groupIdx = Data.groupIdx;
    dataGroups = unique(groupIdx, 'stable');
    % Change the flag to plot each group with all series in each group on
    % an individual axis
    flagSeparateAxes = 0;
    flagGroupAxes = 1;
else
    % Only a single group
    groupIdx = ones(length(y), 1);
    dataGroups = 1;
    flagGroupAxes = 0;
end
% Handle data series line labels:
lineLabels = cell(1,length(dataSeries));
for iSeries = 1:length(dataSeries)
    if flagLabels
        % Grab labels for each data series:
        thisDataSeries = Data(seriesIdx == dataSeries(iSeries),:);
        lineLabels{iSeries} = thisDataSeries.(PlotOpt.DataSeriesLabelVar){1};
    else
        % Create some labels if there aren't any:
        lineLabels{iSeries} = sprintf("Series %d", iSeries);
    end
end
titleLabels = lineLabels; % for individual data series plots
% Organize line labels for plotting all results on the same axis:
lineObjects = gobjects(length(dataSeries), 1);
if strcmpi(PlotOpt.DataSeriesLabelFormat, 'number on plot')
    for iLabel = 1:length(lineLabels)
        lineLabels{iLabel} = sprintf("%d: %s", iLabel, lineLabels{iLabel});
    end
end
    
axes = [];
dataSeriesGlobal = dataSeries;
for iGroup = 1:length(dataGroups)
    % Get the data and data series just from this group
    thisGroup = dataGroups(iGroup);
    mask = groupIdx == thisGroup;
    GroupData = Data(mask, :);
    dataSeries = unique(GroupData.seriesIdx, 'stable');
    
    % Handle data group labels:
    if flagGroupAxes
        if isfield(PlotOpt, 'DataGroupLabelVar')...
                && ~isempty(PlotOpt.DataGroupLabelVar)...
                && any(strcmp(Data.Properties.VariableNames, PlotOpt.DataGroupLabelVar))
            % Grab a label for this group
            labelData = GroupData.(PlotOpt.DataGroupLabelVar);
            if isnumeric(labelData) || ischar(labelData) || isstring(labelData)
                groupLabel = labelData(1);
            elseif iscellstr(labelData)
                groupLabel = labelData{1};
            else
                error('Cannot read DataGroupLabelVar data type, must be numeric, char array, string array, or cellstr.')
            end
        else
            groupLabel = sprintf("Group %d", dataGroups(iGroup));
        end
    end
    
    % If all data series are plotted on the same axis:
    if flagSeparateAxes == 0
        ax1 = nexttile; hold on; box on; grid on;
        axes = [axes; ax1];
    end
    
    % Iterate through the data series:
    for iSeries = 1:length(dataSeries)
        % If each data series gets its own axis:
        if flagSeparateAxes == 2
            ax1 = nexttile; hold on; box on; grid on;
            axes = [axes; ax1];
        end
        % Collect data from this data series:
        thisSeries = dataSeries(iSeries);
        idxSeriesGlobal = find(dataSeriesGlobal == thisSeries);
        mask = seriesIdx == thisSeries;
        xSeries = x(mask);
        ySeries = y(mask);
        
        % Check Data/FitLineProp coloring and style options:
        idxThisSeries = unique(Data.seriesIdx, 'stable') == thisSeries;
        [dataLineSpec, DataLineProperties] = setLineProperties(dataLineColors, dataMarkerColors, idxThisSeries, DataLineProp);
        % Plot the line:
        lineObjects(idxSeriesGlobal) = plot(ax1, xSeries, ySeries, dataLineSpec);
        % If the labels are being put in a legend, add a DisplayName to the
        % DataLineProperties:
        if any(strcmpi(PlotOpt.DataSeriesLabelFormat, {'number on plot','in legend'}))
            DataLineProperties.DisplayName = lineLabels{idxSeriesGlobal};
        end
        % Set line properties:
        set(lineObjects(idxSeriesGlobal), DataLineProperties);
        % Display label text on the plot if it is desired:
        if flagLabelsOnPlot
            drawLabelOnPlot(ax1, xSeries(end), ySeries(end), lineLabels, idxSeriesGlobal, labelColors, PlotOpt)
        end
        % Plot decorations for individual plots:
        if flagSeparateAxes == 2
            % Title for individual plots:
            title(ax1, titleLabels{idxSeriesGlobal})
            % Common plot decorations:
            decoratePlotAxes(ax1, xStr, yStr, PlotOpt)
        end
    end
    % Decorate single axis or group axes plots:
    if flagSeparateAxes == 0
        decoratePlotAxes(ax1, xStr, yStr, PlotOpt)
    end
    % Title for group axes:
    if flagGroupAxes
        title(ax1, groupLabel)
    end
end
% Legend, if it is specified:
if any(strcmpi(PlotOpt.DataSeriesLabelFormat, {'number on plot','in legend'}))
    lgd = legend(lineObjects);
    if isfield(PlotOpt, 'LegendNumColumns') && ~isempty(PlotOpt.LegendNumColumns)
        lgd.NumColumns = PlotOpt.LegendNumColumns;
    end
    if isfield(PlotOpt, 'LegendPosition') && ~isempty(PlotOpt.LegendPosition)
        lgd.Layout.Tile = PlotOpt.LegendPosition; % default is 'west'
    else
        lgd.Layout.Tile = 'west';
    end
end
% Colorbar
if isfield(PlotOpt, 'Colorbar') && strcmpi(PlotOpt.Colorbar, 'On')
    colors = colorBarOpts.Colormap;
    colormap(colors);
    cbar = colorbar('Ticks', colorBarOpts.Ticks,...
        'TickLabels', colorBarOpts.TickLabels,...
        'Limits', colorBarOpts.Limits,...
        'FontWeight', 'bold');
    caxis(colorBarOpts.Limits);
    cbar.Label.String = colorBarOpts.LabelString;
    cbar.Label.Rotation = 90;
    if isfield(PlotOpt, 'ColorbarPosition') && ~isempty(PlotOpt.ColorbarPosition)
        cbar.Layout.Tile = PlotOpt.ColorbarPosition;
    else
        cbar.Layout.Tile = 'east';
    end
end
% Title for a single set of axes
if isfield(PlotOpt, 'Title')
    title(t, PlotOpt.Title)
end
% Link axes:
if isfield(PlotOpt, 'LinkAxes') && strcmpi(PlotOpt.LinkAxes, 'on')
    linkaxes(axes);
end
end

function labelColors = getDataSeriesLabelColors(Data, PlotOpt)
numSeries = length(unique(Data.seriesIdx));
isLabelColorSpecified = isfield(PlotOpt, 'DataSeriesLabelColor') && ~isempty(PlotOpt.DataSeriesLabelColor);
if isLabelColorSpecified
    if isnumeric(PlotOpt.DataSeriesLabelColor)
        if all(size(PlotOpt.DataSeriesLabelColor) == [1 3])
            labelColors = repmat(PlotOpt.DataSeriesLabelColor, numSeries, 1);
        elseif all(size(PlotOpt.DataSeriesLabelColor) == [numSeries 3])
            labelColors = PlotOpt.DataSeriesLabelColor;
        else
            error("Specify either a single RGB DataSeriesLabelColor for all data series, or one RGB color for each data series.")
        end
    elseif ischar(PlotOpt.DataSeriesLabelColor)
        if ~contains(PlotOpt.DataSeriesLabelColor, 'By')
            % If the lineColor isn't specified 'By varStr':
            % Do nothing (color name, shorthand name, or hex code)
            labelColors = 'In PlotOpt';
        else
            assert(~isempty(cmap), "Must provide a colormap to color 'By varStr'.")
            % Get the name of the variable:
            varStr = split(PlotOpt.DataSeriesLabelColor, ' ');
            varStr = varStr{2};
            % Get the color for each data series
            labelColors = colorDataSeriesByVar(Data, varStr, cmap);
        end
    end
else
    labelColors = repmat([0 0 0], numSeries, 1);
end
end

function [flagSeparateAxes, flagLabels, flagLabelsOnPlot] = setFlags(Data, PlotOpt)
if strcmpi(PlotOpt.SeparateAxes, 'By data group')
    flagSeparateAxes = 1;
elseif strcmpi(PlotOpt.SeparateAxes, 'By data series')
    flagSeparateAxes = 2;
else
    flagSeparateAxes = 0;
end
if ~isempty(PlotOpt.DataSeriesLabelVar)...
        && any(strcmpi(Data.Properties.VariableNames, PlotOpt.DataSeriesLabelVar))
    flagLabels = 1;
else
    flagLabels = 0;
end
if any(strcmpi(PlotOpt.DataSeriesLabelFormat, {'label on plot','number on plot'}))
    flagLabelsOnPlot = 1;
else
    flagLabelsOnPlot = 0;
end
end

function [lineSpec, LineProperties] = setLineProperties(lineColors, markerColors, iSeries, LineProperties)
% lineColors is either:
%   1) an array, with a RGB triplet for each data series
%       Action: For each data series, input the triplet into the LineProp
%       field 'Color' before calling 'set'.
%   2) 'In LineSpec', denoting color is input in LineProp.LineSpec
%       Action: Color is provided by the LineSpec. Remove the LineProp
%       field 'Color' before calling 'set'.
%   3) 'In LineProp', denoting color is set in the LineProp.Color field,
%       and 'set' can be called without changing anything.
% markerColors is either:
%   1) an array, with a RGB triplet for each data series
%       Action: For each data series, input the triplet into the LineProp
%       field 'MarkerFaceColor' before calling 'set'.
%   2) 'In LineProp', denoting color is set in the LineProp.MarkerFaceColor
%       field, and 'set' can be called without changing anything.
%   3) 'none', denoting no marker fill color
%       Action: set LineProp.MarkerFaceColor to 'none' before calling 'set'
if isnumeric(lineColors)
    LineProperties.Color = lineColors(iSeries,:);
elseif strcmp(lineColors, 'In LineSpec')
    if isfield(LineProperties, 'Color')
        LineProperties = rmfield(LineProperties, 'Color');
    end
else
    assert(strcmp(lineColors, 'In LineProp'),...
        "LineProperties.Color could not be interpreted correctly when calling 'getDataColors.m' or 'getFitColors.m', check PlotOpt.Data/FitLineProp fields Color and LineSpec for errors.")
end
if isnumeric(markerColors)
    LineProperties.MarkerFaceColor = markerColors(iSeries,:);
else
    assert(any(strcmp(markerColors, {'In LineProp', 'none'})),...
        "LineProperties.MarkerFaceColor could not be interpreted correctly when calling 'getDataColors.m' or 'getFitColors.m', check PlotOpt.Data/FitLineProp field MarkerFaceColor for errors.")
end
if isfield(LineProperties, 'LineSpec')
    lineSpec = LineProperties.LineSpec;
    LineProperties = rmfield(LineProperties, 'LineSpec');
else
    lineSpec = '';
end
end

function drawLabelOnPlot(ax, x, y, lineLabels, iSeries, labelColors, PlotOpt)
if strcmpi(PlotOpt.DataSeriesLabelFormat, 'label on plot')
    labelString = strcat(" ", lineLabels{iSeries});
elseif strcmpi(PlotOpt.DataSeriesLabelFormat, 'number on plot')
    labelString = sprintf(" %d", iSeries);
end
if strcmpi(labelColors, 'In PlotOpt')
    if isfield(PlotOpt, 'DataSeriesLabelFontSize') && isnumeric(PlotOpt.DataSeriesLabelFontSize) && ~isempty(PlotOpt.DataSeriesLabelFontSize)
        text(ax, x, y, labelString, 'Color', PlotOpt.DataSeriesLabelColor, 'FontSize', PlotOpt.DataSeriesLabelFontSize)
    else
        text(ax, x, y, labelString, 'Color', PlotOpt.DataSeriesLabelColor)
    end
else
    if isfield(PlotOpt, 'DataSeriesLabelFontSize') && isnumeric(PlotOpt.DataSeriesLabelFontSize) && ~isempty(PlotOpt.DataSeriesLabelFontSize)
        text(ax, x, y, labelString, 'Color', labelColors(iSeries,:), 'FontSize', PlotOpt.DataSeriesLabelFontSize)
    else
        text(ax, x, y, labelString, 'Color', labelColors(iSeries,:))
    end
end
end

function decoratePlotAxes(ax1, xStr, yStr, PlotOpt)
if isfield(PlotOpt, 'XLabel') && ~isempty(PlotOpt.XLabel)
    xlabel(ax1, (PlotOpt.XLabel));
else
    xlabel(ax1, varStrToAxisLabel(xStr));
end
if isfield(PlotOpt, 'YLabel') && ~isempty(PlotOpt.YLabel)
    ylabel(ax1, (PlotOpt.YLabel));
else
    ylabel(ax1, varStrToAxisLabel(yStr));
end
if isfield(PlotOpt, 'AxisLimits') && ~isempty(PlotOpt.AxisLimits)
    axis(ax1, PlotOpt.AxisLimits);
end
if isfield(PlotOpt, 'xScaleMultiplier') && ~isempty(PlotOpt.xScaleMultiplier) && isnumeric(PlotOpt.xScaleMultiplier)
    % Make the range of the x axis bigger, without changing the
    % lower bound.
    xLimits = xlim(ax1);
    xRange = abs(xLimits(2) - xLimits(1));
    xLimits = [xLimits(1), xLimits(1)+xRange*PlotOpt.xScaleMultiplier];
    xlim(ax1, xLimits);
end
if isfield(PlotOpt, 'AxisType')
    switch PlotOpt.AxisType
        case 'semilogx'
            set(ax1, 'XScale', 'log')
        case 'semilogy'
            set(ax1, 'YScale', 'log')
        case 'loglog'
            set(ax1, 'XScale', 'log')
            set(ax1, 'YScale', 'log')
    end
end
end