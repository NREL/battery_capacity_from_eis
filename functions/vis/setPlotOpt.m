function PlotOpt = setPlotOpt(PlotOpt)
% PlotOpt = SETPLOTOPT(PlotOpt)
% Sets plotting options for plot format, data series label/text
% options, and what plot details to show (residuals plots, confidence
% intervals) using name-value pair inputs. Using this function sets some 
% default options, and provides default values for other fields to ensure 
% error-free plot drawing by the plotFit, plotData, or plotSim functions. 
% Inputs:
%   Name-value pairs corresponding to PlotOpt fields.
% Output:
%   PlotOpt: Struct setting plot formatting options and styles.
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
%       'LegendPosition': 'north', 'south', 'east', or 'west' (always outside)
%       'LegendNumColumns': number of columns for the legend
%       'LegendLabels': Cellstr, used in 'plot_Comparison' functions
%         to label each separate data set.
%       'FitStatsBox': include a text box with the R2, R2adj,
%         and MAE
%       'Colormap': specify which colormap function to use to
%         generate colors, if any of the colors for the data
%         labels, data points, or fit lines are specified as
%         'By xStr'. Accepts any MATLAB colormap function name,
%         as well as colormaps from the 'cmocean' or
%         'brewermap' functions off of the Mathworks File
%         Exchange.
%       'Colorbar': 'On' or 'Off', plots a colorbar if the line or marker
%         colors have been specified as 'By varStr'.
%       'ColorbarPosition': 'north', 'south', 'east', or 'west' (always outside)
%       'ResidualsPlots': 'On' or 'Off' (Uses same AxisType as
%         main plot)
%       'ConfidenceInterval': [lb ub], sent to prctile to
%         calculate lower and upper bounds for the yFit
%         confidence interval. Confidence intervals are shown
%         as shaded regions around yFit, and also shown in
%         residual error plots if these are being plotted.
%       'DataSeriesLabelFormat': If 'IndividualDataSeriesAxes'
%         is 'On', the entries in Data.label are used to title
%         each data series plot. For 'Off' (single plot for all
%         data series), the entries in Data.label are plotted
%         following the rules described below for the following
%         options: 'label on plot', 'number on plot', 'in legend'
%           'off': no labels anywhere
%           'label on plot': Writes the text from each entry of
%             DataSeriesLabels next to the final point of each
%             data series. As labels are on the plot, a legend
%             is not shown for this case.
%           'number on plot': Writes the number of each data
%             series (its unique value of FitResult.seriesIdx)
%             next to the final point of each data series. A
%             legend is shown to the eastoutside of the plot
%             with the labels "number: DataSeriesLabels{number}".
%           'in legend': No text on the plot denoting data
%             series. A legend is shown to the eastoutside of
%             the plot with the labels from DataSeriesLabels.
%       'DataSeriesLabelVar': Table variable to grab labels from.
%       'DataSeriesLabelColor': Color for the data series
%         labels/numbers on the plot and in the legend. Can be
%         any valid MATLAB color ('k',[0 0 0]) or the phrase
%         'By xStr', which colors the label for each data series
%         according to its unique (or average) value of xStr,
%         using the colormap specified in formatOpt.Colormap.
%         Default color is black.
%       'DataLabelFontSize': font size for all data labels
%         printed on the figure.
%       'DataGroupLabelVar': Table variable to grab group labels from. Used
%         title plots if SeparateAxes = 'By data group'.
%       'LinkAxes': sets all separate axes in a figure to have the same x
%         and y  axes (separately considers fit plots and residuals plots).

arguments
    PlotOpt.SeparateAxes char = 'Off'
    PlotOpt.AxisType char = 'plot'
    PlotOpt.AxisLimits double = []
    PlotOpt.xScaleMultiplier double = 1.1
    PlotOpt.XLabel string = []
    PlotOpt.YLabel string = []
    PlotOpt.Title string = []
    PlotOpt.LegendPosition char = 'west'
    PlotOpt.LegendNumColumns double = []
    PlotOpt.LegendLabels cell = {}
    PlotOpt.FitStatsBox char = 'On'
    PlotOpt.Colormap char = []
    PlotOpt.Colorbar char = 'Off'
    PlotOpt.ColorbarPosition = 'east'
    PlotOpt.ResidualsPlots char = 'On'
    PlotOpt.ConfidenceInterval double = []
    PlotOpt.DataSeriesLabelFormat char = 'none'
    PlotOpt.DataSeriesLabelVar char = []
    PlotOpt.DataSeriesLabelColor = 'k' % can also be RGB triplet, or a colormap with a triplet for each data series
    PlotOpt.DataSeriesLabelFontSize double = []
    PlotOpt.DataGroupLabelVar char = []
    PlotOpt.LinkAxes char = 'On'
end
% Check some of the inputs:
if ~isempty(PlotOpt.AxisLimits)
    assert(all(size(PlotOpt.AxisLimits) == [1 4]) && isnumeric(PlotOpt.AxisLimits),...
        "AxisLimits must of the form [xMin xMax yMin yMax].")
end
if ~isempty(PlotOpt.xScaleMultiplier)
    assert(all(size(PlotOpt.xScaleMultiplier) == [1 1]) && isnumeric(PlotOpt.xScaleMultiplier),...
        "xScaleMultiplier must be a single value.")
end
if ~isempty(PlotOpt.ConfidenceInterval)
    assert(all(size(PlotOpt.ConfidenceInterval) == [1 2]) && isnumeric(PlotOpt.ConfidenceInterval),...
        "ConfidenceInterval must of the form [lowerBound upperBound].")
end
if ~isempty(PlotOpt.DataSeriesLabelFontSize)
    assert(all(size(PlotOpt.DataSeriesLabelFontSize) == [1 1]) && isnumeric(PlotOpt.DataSeriesLabelFontSize),...
        "DataSeriesLabelFontSize must be a single value.")
end
% name-value pair options:
assert(any(strcmpi(PlotOpt.SeparateAxes, {'off', 'by data series', 'by data group'})),...
    "SeparateAxes must be 'Off', 'By data series', or 'By data group'.")
assert(any(strcmpi(PlotOpt.LegendPosition, {'east', 'west', 'north', 'south'})),...
    "LegendPosition must be 'east', 'west', 'north', or 'south'.")
assert(any(strcmpi(PlotOpt.ColorbarPosition, {'east', 'west', 'north', 'south'})),...
    "ColorbarPosition must be 'east', 'west', 'north', or 'south'.")
assert(any(strcmpi(PlotOpt.Colorbar, {'on', 'off'})),...
    "Colorbar must be 'On' or 'Off'.")
dataLabelOptions = {'none', 'number on plot', 'label on plot', 'in legend'};
assert(any(strcmpi(PlotOpt.DataSeriesLabelFormat, dataLabelOptions)),...
    "DataSeriesLabelFormat must be 'none', 'number on plot', 'label on plot', or 'in legend'.")
assert(any(strcmpi(PlotOpt.FitStatsBox, {'on', 'off'})),...
    "FitStatsBox must be 'On' or 'Off'.")
assert(any(strcmpi(PlotOpt.ResidualsPlots, {'on', 'off'})),...
    "ResidualsPlots must be 'On' or 'Off'.")
assert(any(strcmpi(PlotOpt.LinkAxes, {'on', 'off'})),...
    "LinkAxes must be 'On' or 'Off'.")
end