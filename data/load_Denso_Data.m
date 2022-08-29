% Paul Gasper, NREL, 2/2022
clear; close all; clc
load('Denso\DensoData_EIS.mat', 'DensoData_EIS')
Data = DensoData_EIS; clearvars DensoData_EIS

% Mask off noisy data at high frequency
freq = Data.Freq(1,:);
maskKeep = freq < 3.5e4;
Data.Freq = Data.Freq(:, maskKeep);
Data.Zreal = Data.Zreal(:, maskKeep);
Data.Zimag = Data.Zimag(:, maskKeep);

% Calculate Z, Zmag, and Zphz
Data.Z = Data.Zreal + 1i.*Data.Zimag;
Data.Zmag = abs(Data.Z);
Data.Zphz = angle(Data.Z).*(180/pi);

% The EIS data table (Data) is missing some variables that we want from the
% raw data table. Load the raw data table, uniquely identify each row in
% each table, and map variables between the two (the raw data table has
% many capacity checkups that do not have EIS measurements).
% Load the raw data table with all measurements
load('Denso\DensoData.mat', 'DensoData');
% Remove rows with NAN time
DensoData = DensoData(~isnan(DensoData.t), :);
% There's a bunch of NAN resistance measurements, but they always appear in
% the middle of each data series, so we can interpolate the values.
DensoData.r25degC = fillmissing(DensoData.r25degC, 'linear');
DensoData.R25degC = fillmissing(DensoData.R25degC, 'linear');
DensoData.rm10degC = fillmissing(DensoData.rm10degC, 'linear');
DensoData.Rm10degC = fillmissing(DensoData.Rm10degC, 'linear');

%{
 % investigate NANs and fillmissing method
[DensoData.seriesIdx, isnan(DensoData.R25degC)];
[a,b] = fillmissing(DensoData.R25degC, 'linear');
figure; hold on; plot(a, 'ok'); plot(a(b), 'xr')
figure; hold on; plot(DensoData.t, a, 'ok'); plot(DensoData.t(b), a(b), 'xr')
c = a; c(b) = 0; [c,a];
%}

% Create a variable that can uniquely identify capacity checkups across the
% two tables (not every capacity value is totally unique, some measurements
% were the same within precision, and the Data table has measurements
% repeated where there were mutiple EIS measurements at a given capacity
% value).
id = Data.seriesIdx .* Data.Qb3;
idRaw = DensoData.seriesIdx .* DensoData.Qb3;
idx = find(idRaw == id');
idx = mod(idx, height(DensoData));
% Copy over the variables of interest
Data.r25C = DensoData.r25degC(idx);
Data.R25C = DensoData.R25degC(idx);
Data.rm10C = DensoData.rm10degC(idx);
Data.Rm10C = DensoData.Rm10degC(idx);
% Reorder variables
Data = removevars(Data, {'rc','Rc'});
Data = renamevars(Data, 'N100', 'EFC');
Data = movevars(Data, 'r25C', 'After', 'Qb3');
Data = movevars(Data, 'R25C', 'After', 'r25C');
Data = movevars(Data, 'rm10C', 'After', 'R25C');
Data = movevars(Data, 'Rm10C', 'After', 'rm10C');

%{
% Interpolate EIS
%{
We have more capacity measurements than we have impedance measurements.
Interpolate impedance to get a synthetic impedance measurement
corresponding to each capacity measurement. There is only enough data to
interpolate at -10C and 25C (only 1 0C and 10C measurement per cell).

It's not obvious what variable we should interpolate the EIS data
against. Plot the value of Zmag@5e-3Hz versus Qb3 and R (at the given temp)
%}
plotInterpVariables(Data)
%{
The impedance is much more correlated with the DC resistance measurement
than it is the capacity, so it's better to interpolate versus the DC
resistance measurement rather than by the capacity.
%}
Data = interpolateData(Data, DensoData);
%}


% There's also data where both SOC and temperature were varied widely, both
% at BOL and at various states of health. These tests were conducted
% out-of-sequence with the rest of the aging study, so they can't be nicely
% integrated with the other test data (the values for time and cycle count,
% t and N, are not recorded). 
load('Denso\DensoData_EIS_vSOCvT.mat', 'DensoData_EIS_vSOCvT')
Data2 = DensoData_EIS_vSOCvT; clearvars DensoData_EIS_vSOCvT
Data2 = renamevars(Data2, {'groupIdx', 'TdegC', 'soc'}, {'seriesIdx', 'TdegC_EIS', 'soc_EIS'});
% Make it so the seriesIdx from the Data and Data2 don't overlap
Data2.seriesIdx = Data2.seriesIdx + max(Data.seriesIdx);
% Mask off noisy data at high frequency
freq = Data2.Freq(1,:);
maskKeep = freq < 3.5e4;
Data2.Freq = Data2.Freq(:, maskKeep);
Data2.Zreal = Data2.Zreal(:, maskKeep);
Data2.Zimag = Data2.Zimag(:, maskKeep);
% Add the Z, Zmag, Zphz variables like for the aging study table.
Data2.Z = Data2.Zreal + 1i.*Data2.Zimag;
Data2.Zmag = abs(Data2.Z);
Data2.Zphz = angle(Data2.Z).*(180/pi);

% The 'vSOC_vT' study has a slightly different EIS protocol; previously,
% EIS was measured down to 0.00501 Hz, with a final end-point at a nice
% clean 0.005 Hz. In the 'vSOC_vT' study, this final end point was not
% recorded. Remove it from the aging data table so that the two data tables
% have the same information. We aren't really losing any information here, 
% since we still have the values for the impedance at 0.00501 Hz. The
% frequency values from the two tables are actually a tiny bit different
% due to this change in protocol, as well, but the difference is less than
% 0.1%, so it's not worth correcting.
Data.Freq = Data.Freq(:, 1:end-1); Data2.Freq = Data2.Freq(:, 1:end-1);
Data.Zreal = Data.Zreal(:, 1:end-1); Data2.Zreal = Data2.Zreal(:, 1:end-1);
Data.Zimag = Data.Zimag(:, 1:end-1); Data2.Zimag = Data2.Zimag(:, 1:end-1);
Data.Z = Data.Z(:, 1:end-1); Data2.Z = Data2.Z(:, 1:end-1);
Data.Zmag = Data.Zmag(:, 1:end-1); Data2.Zmag = Data2.Zmag(:, 1:end-1);
Data.Zphz = Data.Zphz(:, 1:end-1); Data2.Zphz = Data2.Zphz(:, 1:end-1);

% 'Unwrap' all the impedance data into unique column variables
% Zreal, Zimag, Zmag, and Zphz are all just 1 'variable' each. Need to
% give each frequency point a different variable name.
freq = Data.Freq(1,:)';
freqStr = compose("%0.2gHz", freq);
varsZreal = join([repmat("Zreal", length(freq), 1), freqStr], '_');
varsZimag = join([repmat("Zimag", length(freq), 1), freqStr], '_');
varsZmag = join([repmat("Zmag", length(freq), 1), freqStr], '_');
varsZphz = join([repmat("Zphz", length(freq), 1), freqStr], '_');
% Aging study data
x = [Data.Zreal, Data.Zimag, Data.Zmag, Data.Zphz];
EIS = array2table(x, 'VariableNames', [varsZreal; varsZimag; varsZmag; varsZphz]);
DataFormatted = [Data(:, 1:25), EIS];
% vSOC_vT study data
x = [Data2.Zreal, Data2.Zimag, Data2.Zmag, Data2.Zphz];
EIS = array2table(x, 'VariableNames', [varsZreal; varsZimag; varsZmag; varsZphz]);
Data2Formatted = [Data2(:, 1:5), EIS];

save('Data_Denso2021.mat', 'Data', 'DataFormatted', 'Data2', 'Data2Formatted')

%% Helper methods
function Data_ = interpolateData(Data, DataQ)
% Data = table with x and y data
% DataQ = table with xq data (lookup query)
T_EIS = [-10, 25];
unqiueSeries = unique(Data.seriesIdx,'stable');
% Add a variable that tells us whether each row is synthetic or not
Data.isInterpEIS = zeros(height(Data), 1);
Data = movevars(Data, 'isInterpEIS', 'After', 'Rm10C');
for thisSeries = unqiueSeries'
    % get query/fill points: R @ -10,25 for this cell
    DataQ_i = DataQ(DataQ.seriesIdx == thisSeries, :);
    R25C = DataQ_i.R25degC;
    Rm10C = DataQ_i.Rm10degC;
    % Grab other vars we'll have to copy over
    t     = DataQ_i.t;
    N     = DataQ_i.N;
    EFC   = DataQ_i.EFC;
    q     = DataQ_i.q;
    Qb3   = DataQ_i.Qb3;
    r25C  = DataQ_i.r25degC;
    rm10C = DataQ_i.rm10degC;
    % get start/end row indices
    mask = Data.seriesIdx == thisSeries;
    idxSeriesStart = find(mask, 1);
    idxSeriesEnd = find(mask, 1, 'last');
    Data_i = Data(mask, :);
    % for each temp, get x and y data, interpolate
    for T = T_EIS
        maskT = Data_i.TdegC_EIS == T;
        idxTStart = find(maskT, 1);
        idxTEnd = find(maskT, 1, 'last');
        Data_i_T = Data_i(maskT, :);
        % Interpolate using resistance values at -10C (25C DC resistance is
        % extremely noisy)
        xq = Rm10C;
        x = Data_i_T.Rm10C;
%         if T == -10
%             xq = Rm10C;
%             x = Data_i_T.Rm10C;            
%         else %25C
%             xq = R25C;
%             x = Data_i_T.R25C;  
%         end

        % Make sure we only interpolate, not extrapolate (don't
        % make predictions of what the EIS looks like at capacity
        % values lower than the lowest capacity where there is 
        % measured EIS for this cell).
        maskInterp = q >= min(Data_i_T.q);
        xq = xq(maskInterp);
        % Grab the y data (impedance):
        y1 = Data_i_T.Zreal;
        y2 = Data_i_T.Zimag;
        y3 = Data_i_T.Zmag;
        y4 = Data_i_T.Zphz;
        % Interpolate:
        yq1 = interp1(x, y1, xq, 'pchip');
        yq2 = interp1(x, y2, xq, 'pchip');
        yq3 = interp1(x, y3, xq, 'pchip');
        yq4 = interp1(x, y4, xq, 'pchip');
        
        % Create a new table that has the same variables as the old table
        % but with more rows. Fill all the rows with the correct values
        % from the raw data table.
        DataInterp = repmat(Data_i_T(1,:), length(xq), 1);
        DataInterp.t     = t(maskInterp);
        DataInterp.N     = N(maskInterp);
        DataInterp.EFC   = EFC(maskInterp);
        DataInterp.q     = q(maskInterp);
        DataInterp.Qb3   = Qb3(maskInterp);
        DataInterp.r25C  = r25C(maskInterp);
        DataInterp.R25C  = R25C(maskInterp);
        DataInterp.rm10C = rm10C(maskInterp);
        DataInterp.Rm10C  = Rm10C(maskInterp);
        % Fill EIS data vars
        DataInterp.Zreal(:,:) = yq1;
        DataInterp.Zimag(:,:) = yq2;
        DataInterp.Zmag(:,:) = yq3;
        DataInterp.Zphz(:,:) = yq4;
        % fill interp flag
        DataInterp.isInterpEIS(:) = ~any(x == xq');
        
        % Replace the old rows in Data_i with the new data
        Data_i = [Data_i(1:(idxTStart-1),:); DataInterp; Data_i((idxTEnd+1):end,:)];
    end
    % Replace the old rows in Data with Data_i
    Data = [Data(1:(idxSeriesStart-1),:); Data_i; Data((idxSeriesEnd+1):end,:)];
end
Data_ = Data;
end


function plotInterpVariables(Data)
Data_m10C = Data(Data.TdegC_EIS == -10, :);
Data_25C = Data(Data.TdegC_EIS == 25, :);
% Plot
figure; tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
% 25C data
nexttile; axis square
plot(Data_25C.Qb3, Data_25C.Zmag(:,end), 'ok')
xlabel('C/3 discharge capacity @ 25\circC [Ah]'); ylabel('|Z| @ 0.005 Hz, 25\circC [\Omega]');
nexttile; axis square
plot(Data_25C.Rm10C, Data_25C.Zmag(:,end), 'ok')
xlabel('10s avg. DC pulse resistance @ 25\circC [\Omega]'); ylabel('|Z| @ 0.005 Hz, -10\circC [\Omega]');
% m10C data
nexttile; axis square
plot(Data_m10C.Qb3, Data_m10C.Zmag(:,end), 'ok')
xlabel('C/3 discharge capacity @ 25\circC [Ah]'); ylabel('|Z| @ 0.005 Hz, -10\circC [\Omega]');
nexttile; axis square
plot(Data_m10C.Rm10C, Data_m10C.Zmag(:,end), 'ok')
xlabel('10s avg. DC pulse resistance @ -10\circC [\Omega]'); ylabel('|Z| @ 0.005 Hz, -10\circC [\Omega]');
% formatting
set(gcf, 'Units', 'inches', 'Position', [5,2.989583333333333,6.5,5]);
annotation(gcf,'textbox',...
    [0.0826391941391941 0.884523809523811 0.0400714285714286 0.0642857142857147],...
    'String','a',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.563408424908425 0.878273809523811 0.0400714285714285 0.0642857142857147],...
    'String','b',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.0778315018315018 0.388690476190477 0.0400714285714286 0.0642857142857147],...
    'String','c',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
annotation(gcf,'textbox',...
    [0.565010989010989 0.382440476190478 0.0400714285714286 0.0642857142857148],...
    'String','d',...
    'FontSize',12,...
    'FitBoxToText','off',...
    'EdgeColor','none');
end

%{
% Load Denso data
load('Denso\DensoData.mat', 'DensoData')
%{
Process data:
    Keep only data columns of interest
    Remove noisy high-frequency EIS data
    Extract EIS data into table variables
    Remove a noisy EIS measurement from the data set (2nd to last
        measurement on WLTP cell 2)
%}
Data = reassembleDensoData(DensoData);
save('Data_Denso2021.mat', 'Data');




function Data = reassembleDensoData(Data)
% Pull out an initial EIS data file to get the freq EIS_25C data
eis = Data.EIS_25C{1};
freq = eis.Freq;
% Get rid of columns we don't need
varsKeep = {'seriesIdx','testType','name','TdegC','TdegK','soc','Ua','Uc','Voc','dod','Cchg','Cdis','dutycyc','t','N','EFC','q','Qb3','Qb20','EIS_25C'};
Data = Data(:, varsKeep);
% Mask out the BOL data
Data = Data(Data.t ~= 0, :);
% Mask off cell 29 (other than BOL, there is only one EIS measurement,
% which isn't really enough data to make predictions)
Data = Data(Data.seriesIdx ~= 29, :);
% Extract EIS measurements
eis = zeros(height(Data), numel(freq)*2); %Zreal, ZImag
for i = 1:height(Data)
    ThisEIS = Data.EIS_25C{i};
    % Not all RPTs have EIS
    if isempty(ThisEIS)
        eis(i,:) = nan(1, numel(freq)*2);
    else
        eis(i,:) = [ThisEIS.Z_real' ThisEIS.Z_imag'];
    end
end
maskEmptyEIS = any(isnan(eis),2);
eis = eis(~maskEmptyEIS, :);
Data = Data(~maskEmptyEIS, :);
% There's a huge amount of noise at high frequency, remove these data points.
maskKeep = [freq; freq] <= 1e5;
eis = eis(:, maskKeep);
freq = freq(freq <= 1e5);
% Name variables according to frequency
freqStr = compose("%0.2gHz", freq); freqStr(end-1) = "0.00501Hz";
varsZReal = join([repmat("ZReal", length(freq), 1), freqStr], '_');
varsZImag = join([repmat("ZImag", length(freq), 1), freqStr], '_');
% create new table with all the new variables
EIS = array2table(eis, 'VariableNames', [varsZReal; varsZImag]);
Data = [Data, EIS];
% The second to last EIS measurement (WLTP Cell 2) is shifted dramatically 
% to higher Z_real that the other measurements from that cell.
% Remove this measurement, as this sudden increase in Ohmic resistance
% doesn't look like any of the other trends in the data set, so it's
% probably a measurement error.
Data = Data([1:height(Data)-2, height(Data)],:);
end
%}