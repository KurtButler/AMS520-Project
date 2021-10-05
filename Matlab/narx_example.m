
close all;

x; % signal
L; % prediction length

X = embed( x(1:end-L) , Q, tau);
Y = x(1+L:end);

net = feedforwardnet([50,30,10]);
net = train(net, X,Y);

plot( (1:numel(x))',x, ...
      (1:numel(Y))'+L, xp);