
close all;

%% Import data
import_gas;
x = natgas.NXT_CNG_STK;

%% Embedding delay
a = autocorr(x);
tau = find(a<=0.5, 1 ) -1

%% Embedding dimension
rho = 10;
FNNtol = 0.08;
Q = 1;
FNNflag = false;
while ~FNNflag
    Q = Q + 1;
    
    if Q*tau > 0.25*numel(x)
        error('False-nearest-neighbors algorithm failed to converge')
    end
    
    M1 = embed(x,Q,tau);
    M2 = embed(x,Q+1,tau);
    % Make sure that these guys are the same size
    M1 = M1(1:size(M2,1),:);
    FNN = zeros(size(M1,1),1);
    for n = 1:size(M1,1)
        [r,id] = mink( vecnorm(M1-M1(n,:),2,2) ,2);
        Rd = norm(M1(id(2),:)-M1(n,:),2)/sqrt(Q);
        FNN(n) = norm(M2(n,:) - M2(id(2),:),2) > rho*Rd;
    end
    
    if mean(FNN) < FNNtol
        FNNflag = true;
    end
end

Q 


%% Phase portrait
figure
subplot(1,2,1)
plot(x);
grid on;
title('Raw signal')

subplot(1,2,2)
M = embed(x,3,tau);
plot3(M(:,1), M(:,2), M(:,3));
grid on;
title(sprintf('Phase portait\ntau=%d, Q=%d',tau,Q))