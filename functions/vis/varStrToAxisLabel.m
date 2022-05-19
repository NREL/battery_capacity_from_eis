function labelstr = varStrToAxisLabel(varStr)
%labelstr = VARSTRTOAXISLABEL(varStr)
% Uses a switch to take common variable string names
% (shorthand, ex., qdis, N, EFC) and returns a nicely formatted
% axis label string (ex., 'Relative discharge capacity',
% 'Cycles', 'Equivalent Full Cycles').
switch varStr
    case 'Cchg'
        labelstr = 'C_{chg} (hr^{-1})';
    case 'Cdis'
        labelstr = 'C_{dis} (hr^{-1})';
    case 'Crate'
        labelstr = 'Avg. C rate (hr^{-1})';
    case 'Eb20'
        labelstr = 'C/20 energy (Wh)';
    case 'Eb3'
        labelstr = 'C/3 energy (Wh)';
    case 'N100'
        labelstr = 'N_{100} (equivalent full cycles)';
    case 'OFSb20'
        labelstr = 'C/20 dQdV Offset (Ah)';
    case 'OFSb3'
        labelstr = 'C/3 dQdV Offset (Ah)';
    case 'dOFSb20'
        labelstr = 'C/20 dQdV \DeltaOffset (Ah)';
    case 'dOFSb3'
        labelstr = 'C/3 dQdV \DeltaOffset (Ah)';
    case 'QNEb20'
        labelstr = 'C/20 dQdV Neg. Sites (Ah)';
    case 'QNEb3'
        labelstr = 'C/3 dQdV Neg. Sites (Ah)';
    case 'QPEb20'
        labelstr = 'C/20 dQdV Pos. Sites (Ah)';
    case 'QPEb3'
        labelstr = 'C/3 dQdV Pos. Sites (Ah)';
    case 'Qb20'
        labelstr = 'C/20 rel. capacity';
    case 'Qb3'
        labelstr = 'C/3 rel. capacity';
    case 'Rc'
        labelstr = 'R_{charge} (m\Omega at 25^oC, 50%soc)';
    case 'Rc_25oC_20soc'
        labelstr = 'R_{charge} (m\Omega at 25^oC, 20%soc)';
    case 'Rc_25oC_50soc'
        labelstr = 'R_{charge} (m\Omega at 25^oC, 50%soc)';
    case 'Rc_25oC_70soc'
        labelstr = 'R_{charge} (m\Omega at 25^oC, 70%soc)';
    case 'Rc_m10oC_50soc'
        labelstr = 'R_{charge} (m\Omega at -10^oC, 50%soc)';
    case 'Rd_25oC_20soc'
        labelstr = 'R_{discharge} (m\Omega at 25^oC, 20%soc)';
    case 'TdegC'
        labelstr = 'T (\circC)';
    case 'TdegK'
        labelstr = 'T (K)';
    case 'TmeasC'
        labelstr = 'T_{meas} (\circC)';
    case 'TrptC'
        labelstr = 'T_{RPT} (\circC)';
    case 'TrptK'
        labelstr = 'T_{RPT} (K)';
    case 'Ua'
        labelstr = 'U_a (V)';
    case 'UaqT'
        labelstr = 'U_a/T (V/K)';
    case 'Uc'
        labelstr = 'U_c (V)';
    case 'UcqT'
        labelstr = 'U_c/T (V/K)';
    case 'Voc'
        labelstr = 'V_{oc} (V)';
    case 'VocqT'
        labelstr = 'V_{oc}/T (V/K)';
    case 'cellname'
        labelstr = 'cellname';
    case 'date'
        labelstr = 'date';
    case 'dod'
        labelstr = 'DOD';
    case 'dqdt'
        labelstr = 'dqdt (day^{-1})';
    case 'dutycyc'
        labelstr = 'Duty Cycle = cycling time / total time';
    case 'invTdegK'
        labelstr = '1/T (K^{-1})';
    case 'invTrptK'
        labelstr = '1/T_{RPT} (K^{-1})';
    case 'name'
        labelstr = 'name';
    case 'q'
        labelstr = 'Rel. capacity';
    case 'rc'
        labelstr = 'Rel. resistance';
    case 'cb'
        labelstr = 'Rel. conductivity';
    case 'soc'
        labelstr = 'SOC';
    case {'t', 't_day'}
        labelstr = 'Time (days)';
    case 't_years'
        labelstr = 'Time (years)';
    case 'N'
        labelstr = 'N (cycles)';
    case 'Ah'
        labelstr = 'Cumulative energy thoughput (Ah)';
    case 'Wh'
        labelstr = 'Cumulative power thoughput (Wh)';
    case 'R25degC'
        labelstr = 'Avg. resistance at 25^oC (m\Ohms)';
    case 'Rm10degC'
        labelstr = 'Avg. resistance at -10^oC (m\Ohms)';
    case {'qdis', 'qDis'}
        labelstr = 'Relative discharge capacity';
    case 'Qdis'
        labelstr = 'Discharge capacity (Ah)';
    case 'R50'
        labelstr = 'DC resistance at 50% SOC (\Omega)';
    case 'r50'
        labelstr = 'Rel. DC resistance at 50% SOC';
    case 'qloss'
        labelstr = 'Relative capacity loss';
    case 't_hrs'
        labelstr = 'Time (hours)';
    case 'FCE'
        labelstr = 'Full cycle equivalents';
    case 'EFC'
        labelstr = 'Equivalent full cycles';
    case 'rdc'
        labelstr = 'Rel. DC resistance';
    case 'mae'
        labelstr = 'Mean absolute error';
    case 'mape'
        labelstr = 'Mean absolute percent error';
    case 'msd'
        labelstr = 'Mean signed deviation';
    case 'mse'
        labelstr = 'Mean squared error (Chi-squared)';
    case 'rmse'
        labelstr = 'Root mean squared error';
    case 'r2'
        labelstr = 'R^2';
    case 'r2adj'
        labelstr = 'R^2_{adj}';
    case 'dof'
        labelstr = 'Degrees of freedom';
    case 'deltaVeff'
        labelstr = 'Voltage slippage (\DeltaV)';
    case 'cdc'
        labelstr = 'Rel. DC conductivity';
    case 'Cdc'
        labelstr = 'DC conductivity (\Omega^{-1})';
    otherwise
        % Add a \ in front of any underscore or carat so the TeX
        % interpreter doesn't mangle the variable name
        varStr = strrep(varStr, '_', '\_');
        varStr = strrep(varStr, '^', '\^');
        labelstr = varStr;
end
end