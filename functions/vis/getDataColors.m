function [lineColors, markerColors, colorBarOpts] = getDataColors(Data, LineProperties, cmap)
%[lineColors, markerColors, colorBarOpts] = GETDATACOLORS(Data, LineProperties, colormap)
%  Returns a rgb color for the line/marker edges and marker faces for each 
%  unique data series in Data (denoted by a unique value of seriesIdx). The
%  colors can be given in LineProperties.LineSpec in MATLAB shorthand
%  format (color for all data series with LineSpec '-ok' is black), can be
%  specified in MATLAB shorthand format or as RGB values in the fields
%  Color and MarkerFaceColor, or specified as 'By varStr', which colors
%  each data series according to its average value of the variable 'varStr'
%  from the Data table using the input colormap function.
%  Inputs:
%   Data (table): Input data table.
%   LineProperties (struct): Struct specifying options for line properties,
%     with the additional field 'LineSpec', which is input into the plot
%     function.
%   cmap (char vector): Name of a function that generates a colormap,
%     either a MATLAB function, or a cmocean or brewermap colormap.
%  Outputs:
%   lineColors (array): RGB values for the line of each data series.
%   markerColors (array): RGB values for the marker face of each data
%     series.
%   colorBarOpts (struct): Specs for displaying a colorbar.

numSeries = length(unique(Data.seriesIdx));
lineSpec = LineProperties.LineSpec;
% Set the lineColors:
isColorSpecified = isfield(LineProperties, 'Color') && ~isempty(LineProperties.Color);
if contains(lineSpec, {'y','m','c','r','g','b','w','k'})
    assert(~isColorSpecified,...
        'Do not specify data line color in both LineSpec and Color fields.')
    lineColors = 'In LineSpec';
    colorBarOpts = [];
elseif isColorSpecified
    % Color is specified in 'Color' field of LineProperties. If it is a
    % numeric array, repeat the row numSeries times. If it is another valid
    % 'Color' input (color name, shorthand names, hex code), don't do
    % anything. If it is of the form 'By varStr', generate colors from the
    % colormap according to the average value of varStr for each data
    % series from the Data table. 
    if isnumeric(LineProperties.Color)
        if all(size(LineProperties.Color) == [1 3])
            lineColors = repmat(LineProperties.Color, numSeries, 1);
        elseif all(size(LineProperties.Color) == [numSeries 3])
            lineColors = LineProperties.Color;
        else
            error("Specify either a single RGB Color for all data series, or one RGB color for each data series.")
        end
        colorBarOpts = [];
    elseif ischar(LineProperties.Color)
        if ~contains(LineProperties.Color, 'By')
            % If the lineColor isn't specified 'By varStr':
            % Do nothing (color name, shorthand name, or hex code)
            lineColors = 'In LineProp';
            colorBarOpts = [];
        else
            assert(~isempty(cmap), "Must provide a colormap to color 'By varStr'.")
            % Get the name of the variable:
            varStr = split(LineProperties.Color, ' ');
            varStr = varStr{2};
            % Get the color for each data series
            [lineColors, colorBarOpts] = colorDataSeriesByVar(Data, varStr, cmap);
        end
    end
else
    % Assume we just want a different color for each data series, like
    % default MATLAB plotting.
    % If nothing is specified, use the colormap specified in
    % cmap, default to 'Set1' (nice brewermap colormap)
    if isempty(cmap)
        cmap = 'Set1';
    end
    [lineColors, colorBarOpts] = colorDataSeriesByVar(Data, 'seriesIdx', cmap);
end
% Set the marker color:
isMarkerColorSpecified = isfield(LineProperties, 'MarkerFaceColor') && ~isempty(LineProperties.MarkerFaceColor);
if isMarkerColorSpecified
    % Color is specified in 'MarkerFaceColor' field of LineProperties. If 
    % it is a numeric array, repeat the row numSeries times. If it is 
    % another valid 'Color' input (color name, shorthand names, hex code, 
    % 'auto'), don't do anything. If it is of the form 'By varStr', 
    % generate colors from the colormap according to the average value of 
    % varStr for each data series from the Data table.
    if isnumeric(LineProperties.MarkerFaceColor)
        if all(size(LineProperties.MarkerFaceColor) == [1 3])
            markerColors = repmat(LineProperties.MarkerFaceColor, numSeries, 1);
        elseif all(size(LineProperties.MarkerFaceColor) == [numSeries 3])
            markerColors = LineProperties.MarkerFaceColor;
        else
            error("Specify either a single RGB MarkerFaceColor for all data series, or one RGB color for each data series.")
        end
    elseif ischar(LineProperties.MarkerFaceColor)
        if ~contains(LineProperties.MarkerFaceColor, 'By')
            % If the lineColor isn't specified 'By varStr':
            % Do nothing (color name, shorthand name, hex code, or 'auto')
            markerColors = 'In LineProp';
        else
            assert(~isempty(cmap), "Must provide a colormap to color 'By varStr'.")
            % Get the name of the variable:
            varStrMarker = split(LineProperties.MarkerFaceColor, ' ');
            varStrMarker = varStrMarker{2};
            if exist('varStr', 'var')
                assert(strcmp(varStr, varStrMarker),...
                    "If coloring both the line and marker by a variable/fit statistic, they must be the same variable/fit statistic.")
                markerColors = lineColors;
                % colorBarOpts has already been created, no need to change
                % anything
            end
            % Get the color for each data series
            [markerColors, colorBarOpts] = colorDataSeriesByVar(Data, varStrMarker, cmap);
        end
    end
else
    markerColors = 'none';
end
% Return empty colorBarOpts if it hasn't been created.
if ~exist('colorBarOpts', 'var')
    colorBarOpts = [];
end
end