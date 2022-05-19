function LineProp = setLineProp(LineSpec, LineProperties)
% LineProp = SETLINEPROP(LineSpec, LineProperties)
%  Creates a struct storing the properties to use for plotting lines. The
%  input 'lineSpec' corresponds to the 'LineSpec' input to the plot
%  function, and is passed directly to plot (ex., '-ok', 'sr', '--')
%  The line property 'Color' can also be specified differently than normal.
%  To avoid any confusion, do not specify line color(s) in both the
%  LineSpec as well as in LineProperties.Color (this will throw an error
%  later to warn the user).
%   Color: In addition to normal inputs ('r', 'red', RGB triplet, HEX
%   code), color can also be specifed as 'By varStr' (for data and fit
%   lines) or 'By LocalFitStatistic' (for fit lines). Data series are then 
%   colored using the colormap specified in PlotOpt.Colormap (see the 
%   function setPlotOpt) according to their average value of the variable 
%   'varStr' in the Data table, or the average value of the specified 
%   LocalFitStatistic (either 'mae', 'mape', 'mse', 'rmse', 'msd', 'r2',
%   'r2adj', or 'dof').
%  The field DisplayName is ignored, as this is set based on 
%  DataSeriesLabelFormat options (see setPlotOpt).
%  All other line properties are simply passed into the function set() 
%  after plotting each fit/data line, and are explained in the MATLAB
%  documentation.

arguments
    LineSpec char = '-'
    LineProperties.?matlab.graphics.chart.primitive.Line
    LineProperties.Color = ''
end
LineProp = LineProperties;
LineProp.LineSpec = LineSpec;
if isfield(LineProp, 'DisplayName')
    LineProp = rmfield(LineProp, 'DisplayName');
end
if isempty(LineProp.Color)
    LineProp = rmfield(LineProp, 'Color');
end
end
