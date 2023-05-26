clear; clc;
load('data\Faraji-Niri_2023\WholeDataRealSOH.mat')

Data = WholeDataRealSOH; clearvars WholeDataRealSOH
% Numeric data is in cells for some reason. Reformat.
seriesIdx = [Data.Cell_Name{:}]';
q = [Data.SoH_Actual{:}]'./10000; % SoH is scaled by 1e5?
TdegC = [Data.Temp{:}]';
soc = [Data.SoC{:}]';

Data_ = table(seriesIdx, q, TdegC, soc);

% EIS data
% Create variable names
eis_dummy = Data.EIS{1};

% Extract all eis data into a table
for i = 1:height(Data)
    eis_row = Data.EIS{i};
    if i == 1
        eis = [eis_row(:,2); eis_row(:,3)]';
    else
        eis = [eis; [eis_row(:,2); eis_row(:,3)]'];
    end
end

freq = eis_row(:,1);
freqStr = compose("%0.2gHz", freq);
varsZReal =  join([repmat("ZReal",  length(freq), 1), freqStr], '_');
varsZImag =  join([repmat("ZImag",  length(freq), 1), freqStr], '_');
variableNames = [varsZReal; varsZImag];

eis = array2table(eis, 'VariableNames', variableNames);

Data = [Data_, eis];
save("data\Data_FarajiNiri.mat", 'Data');