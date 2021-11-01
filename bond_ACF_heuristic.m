close all;
% import_bond

% Data matrix
X = table2array(bond);
X(:,1) = []; % Get rid of row indices

% Detrending
X = X - sgolayfilt(X, 2, 91);

% Names of features
names =  {"TSY SHORT YLD","TSY LONG YLD","CORP YLD","JUNK YLD","BOND FUT VOL",...
         "STK DX TOT RET","STK IDX VOL","ROW STK TOT RET","CRUDE","CRUDE VOL",...
         "USD","TSY FUT NXT DAY RET"};

% Plot the result
for p = 1:size(X,2)
    subplot(size(X,2),1,p);
    autocorr( X(:,p),30)
    title(names{p})
    ylabel('')
end
sgtitle('ACF of detrended features')
