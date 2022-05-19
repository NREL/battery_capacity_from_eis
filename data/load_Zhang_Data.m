% Load all of the EIS data from the 25C cycled cells from EIS recorded at
% 'state V' (100% SOC after 15 minutes rest post-CCCV charge)
close all; clc; clear;
filepathEis = 'Zhang2020\EIS\EIS_state_V_25C';
filepathCapacity = 'Zhang2020\capacity\Data_Capacity_25C';
cell = {'01','02','03','04','05','06','07','08'};
for iCell = 1:length(cell)
    % Load the capacity measurement for this cell
    capacity = loadCapacityData([filepathCapacity cell{iCell} '.txt'], iCell);
    relCapacity = capacity./capacity(1);
    % Load the EIS data for this cell
    Eis = loadEisData([filepathEis cell{iCell} '.txt']);
    % Number of cycles of capacity/EIS data does not match between files.
    % There are different numbers of capacity/EIS measurements for every
    % cell, it is not consistent. So, we just will grab as many
    % measurements as we can where we have both EIS and capacity data.
    maxCycles = min(height(Eis), length(capacity));
    Eis = Eis(1:maxCycles, :);
    % Store and reorganize table variables.
    Eis.seriesIdx(:) = iCell;
    Eis.Q = capacity(1:maxCycles); Eis.q = relCapacity(1:maxCycles);
    Eis = movevars(Eis, {'seriesIdx', 'cycle', 'Q', 'q'}, 'Before', 1);
    % Store all the data in a table
    if iCell == 1
        Data = Eis;
    else
        Data = [Data; Eis];
    end
end

function capacity = loadCapacityData(filepath, iCell)
switch iCell
    case {1,5}
        opts = delimitedTextImportOptions("NumVariables", 4);
        
        % Specify range and delimiter
        opts.DataLines = [2, Inf];
        opts.Delimiter = "\t";
        
        % Specify column names and types
        opts.VariableNames = ["Var1", "cycleNumber", "oxred", "CapacitymAh"];
        opts.SelectedVariableNames = ["cycleNumber", "oxred", "CapacitymAh"];
        opts.VariableTypes = ["char", "double", "double", "double"];
        
        % Specify file level properties
        opts.ExtraColumnsRule = "ignore";
        opts.EmptyLineRule = "read";
        
        % Specify variable properties
        opts = setvaropts(opts, "Var1", "WhitespaceRule", "preserve");
        opts = setvaropts(opts, "Var1", "EmptyFieldRule", "auto");
    case {2,3,4,6,7,8}
        opts = delimitedTextImportOptions("NumVariables", 6);
        
        % Specify range and delimiter
        opts.DataLines = [2, Inf];
        opts.Delimiter = "\t";
        
        % Specify column names and types
        opts.VariableNames = ["Var1", "cycleNumber", "oxred", "Var4", "Var5", "CapacitymAh"];
        opts.SelectedVariableNames = ["cycleNumber", "oxred", "CapacitymAh"];
        opts.VariableTypes = ["char", "double", "double", "char", "char", "double"];
        
        % Specify file level properties
        opts.ExtraColumnsRule = "ignore";
        opts.EmptyLineRule = "read";
        
        % Specify variable properties
        opts = setvaropts(opts, ["Var1", "Var4", "Var5"], "WhitespaceRule", "preserve");
        opts = setvaropts(opts, ["Var1", "Var4", "Var5"], "EmptyFieldRule", "auto");
end
% Import the data
data = readtable(filepath, opts);

% Extract capacity for each curve
% oxred = 1: charge, oxred = 0: discharge
data = data(data.oxred == 0, :);
cycle = unique(data.cycleNumber);
cycleMask = double(data.cycleNumber == cycle');
capacity = data.CapacitymAh .* cycleMask;
capacity = transpose(max(capacity));
end

function Eis = loadEisData(filepath)
% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 7);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = "\t";

% Specify column names and types
opts.VariableNames = ["times", "cycleNumber", "freqHz", "ReZOhm", "ImZOhm", "ZOhm", "PhaseZdeg"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
EisRaw = readtable(filepath, opts);

% Organize the table into one row per cycle, with each measured
% property (Zreal, Zimag, Z, Phase) getting one column per frequency,
% and one column for the cycle count.
cycle = unique(EisRaw.cycleNumber);
freq = unique(EisRaw.freqHz, 'stable');
eis = zeros(length(cycle), 1+length(freq)*4);
for thisCycle = cycle'
    maskThisCycle = EisRaw.cycleNumber == thisCycle;
    rawValues = [EisRaw.ReZOhm(maskThisCycle); EisRaw.ImZOhm(maskThisCycle); EisRaw.ZOhm(maskThisCycle); EisRaw.PhaseZdeg(maskThisCycle)];
    eis(thisCycle,:) = transpose([thisCycle; rawValues]);
end

% Create variable names
freqStr = compose("%0.2gHz", freq);
varsZReal =  join([repmat("ZReal",  length(freq), 1), freqStr], '_');
varsZImag =  join([repmat("ZImag",  length(freq), 1), freqStr], '_');
varsZ =      join([repmat("Z",      length(freq), 1), freqStr], '_');
varsZPhase = join([repmat("ZPhase", length(freq), 1), freqStr], '_');
variableNames = ["cycle"; varsZReal; varsZImag; varsZ; varsZPhase];

% Create the EIS table
Eis = array2table(eis, 'VariableNames', variableNames);
% Create new variables for each measure, but relative to their values
% at beginning of life
varszReal =  join([repmat("zReal",  length(freq), 1), freqStr], '_');
varszImag =  join([repmat("zImag",  length(freq), 1), freqStr], '_');
varsz =      join([repmat("z",      length(freq), 1), freqStr], '_');
varszPhase = join([repmat("zPhase", length(freq), 1), freqStr], '_');
variableNames = [varszReal; varszImag; varsz; varszPhase];
eisRelative = Eis{:, [varsZReal; varsZImag; varsZ; varsZPhase]} ./ Eis{1, [varsZReal; varsZImag; varsZ; varsZPhase]};
EisRelative = array2table(eisRelative, 'VariableNames', variableNames);
Eis = [Eis, EisRelative];

% Add a frequency variable so that all the EIS information is stored in
% the table
Eis.Freq = repmat(freq', height(Eis), 1);
end