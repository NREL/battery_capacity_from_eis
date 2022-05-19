clear all; close all

%% Convert First Table to csv
load('data/Data_Denso2021.mat')

write(DataFormatted, 'DensoData_EIS.csv')

%% Convert Second Table to csv
write(Data2Formatted, 'DensoData_EIS_vSOCvT.csv')

%% Convert First Table to csvs for impedance.py
for i=1:height(Data)
    SaveName = strcat('impedance csvs/DensoData_EIS/seriesIdx',num2str(Data.seriesIdx(i)),...
        't', num2str(Data.t(i)), 'N',num2str(Data.N(i)),...
        'T',num2str(Data.TdegC_EIS(i)),'soc',num2str(Data.soc_EIS(i)*100),...
        '_', num2str(i),'.csv');
    SaveData = [Data.Freq(i,:)' Data.Zreal(i,:)' Data.Zimag(i,:)' Data.Zmag(i,:)' Data.Zphz(i,:)'];
    writematrix(SaveData, SaveName)
end

%% Convert Second Table to csvs for impedance.py
for i=1:height(Data2)
    SaveName = strcat('impedance csvs/DensoData_EIS_vSOCvT/groupIdx',num2str(Data2.seriesIdx(i)),...
        'T',num2str(Data2.TdegC_EIS(i)),'soc',num2str(Data2.soc_EIS(i)*100),...
        '_', num2str(i),'.csv');
    SaveData = [Data2.Freq(i,:)' Data2.Zreal(i,:)' Data2.Zimag(i,:)' Data2.Zmag(i,:)' Data2.Zphz(i,:)'];
    writematrix(SaveData, SaveName)
end