function [Fitted_Fre,Fitted_Rsquare] = SineWaveFitting(fittingRange,Data)

% Data, n*2, n is the number of time points; the first column is the time (s), the second column is the
% dependent variable

% fittingRange is the frequency range of fitting

% Fitted_Fre; the best fitting frequency in Hz; might be local optimization
% Fitted_Rsquare; the rsqure of fitting 

f_low = fittingRange(1); % lowest frequency can be fitted (Hz)
f_up = fittingRange(2); % highest frequency can be fittted (Hz)

% fitting
options = fitoptions('fourier1','Lower',[-inf -inf -inf f_low*2*pi],...
    'Upper',[inf inf inf f_up*2*pi]);

[fit_object, goodness_fit] = fit(Data(:,1), Data(:,2),'fourier1',options);
parameter_fit = coeffvalues(fit_object);
Fitted_Fre = parameter_fit(4)/(2*pi);
Fitted_Rsquare = goodness_fit.rsquare;