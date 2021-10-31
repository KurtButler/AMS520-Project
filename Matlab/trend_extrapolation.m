
close all;

if ~exist('data','var')
    if exist('C:\Users\kurtb\Documents\MATLAB\AMS520-Project\Matlab','dir')
        addpath(genpath('C:\Users\kurtb\Documents\MATLAB\AMS520-Project\Matlab'))
    end
    if exist('/Users/kbutler/Documents/MATLAB/AMS Project','dir')
        addpath(genpath('/Users/kbutler/Documents/MATLAB/AMS Project'))
    end
    addpath C:\Users\kurtb\Documents\MATLAB\topology
    import_gas;
    % Select some features
    % For now I'm gonna use the easy stuff
    data = [natgas.STOCKS,natgas.JFKTEMP, natgas.CLTTEMP, natgas.ORDTEMP, natgas.HOUTEMP, natgas.LAXTEMP, natgas.NXT_CNG_STK];
    weeks = (1:size(data,1))';
end

% truncate (for testing)
data = data(1:200,:);
weeks = weeks(1:200,:);


% Normalize
data = normalize(data);


% Detrending
datatrend = sgolayfilt( data, 3, 11);
datadetrended = data - datatrend;

% Trend predictions
trendp = zeros(size(datatrend,1),1);
for n = 11:size(trendp,1)
    segment = datatrend(n-4:n-1,end);
%     H = ((1:numel(segment))').^(0:1);
%     b = pinv(H)*segment;
%     yp = ((numel(segment)+1).^(0:1))*b;
%     gp = fitrgp((1:numel(segment))',segment,'BasisFunction','linear');
%     yp = predict(gp,numel(segment)+1);
    func = fit( (1:numel(segment))', segment, 'lowess');
    trendp(n) =  yp;
end


plot(weeks,data(:,end),'k', weeks,datatrend(:,end),'g',weeks,trendp,'r','LineWidth',1.5);
grid on
legend('Target','S-G trend','1-step predictions','Location','best')

