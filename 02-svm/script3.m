%% Same problem with n = 500, bandwidth = 0.05

% SVM non-linéaire
clear; close all;

%% Generate the data 
n = 500;
[Xapp,yapp,Xtest,ytest] = dataset_KM('checkers', n, n^2);
[n, p] = size(Xapp);

figure(1); hold on
% Les labels 1
h1=plot(Xapp(yapp==1,1),Xapp(yapp==1,2),'+r','LineWidth',2);
% Les labels -1
h2=plot(Xapp(yapp==-1,1),Xapp(yapp==-1,2),'db','LineWidth',2);

%% Compute the C-SVM with gaussian kernel (kerneloption = 0.05)
D = (Xapp*Xapp');
N = diag(D);
D = -2*D + N*ones(1,n) + ones(n,1)*N';
kerneloption = .05;
s = 2*kerneloption^2;
K = (exp(-D/s));
G = (yapp*yapp').*K;

%% Same, but with SVM kernel function from the toolbox
kernel = 'gaussian';
K =svmkernel(Xapp,kernel,kerneloption);
G = (yapp*yapp').*K;

C = 10000;
e = ones(n,1);

%% Same, but using monqp script
tic
lambda = eps^.5;
[alpha,b,pos] = monqp(G,e,yapp,0,C,lambda,0);
monqp_time = toc;

%% Nice plot
% Grid
[xtest1 xtest2] = meshgrid([-1:.01:1]*3,[-1:0.01:1]*3);

% Generate 2-dim test vector based on the grid
nn = length(xtest1);
Xgrid = [reshape(xtest1 ,nn*nn,1) reshape(xtest2 ,nn*nn,1)];
Kgrid = svmkernel(Xgrid,kernel,kerneloption,Xapp(pos,:));
ypred = Kgrid*(yapp(pos).*alpha) + b;

% Reshape the solution to the grid shape
ypred = reshape(ypred,nn,nn);

% Plot
contourf(xtest1,xtest2,ypred,50); shading flat;
hold on;
[cc,hh]=contour(xtest1,xtest2,ypred,[-1 0 1],'k');
clabel(cc,hh);
set(hh,'LineWidth',2);
h1=plot(Xapp(yapp==1,1),Xapp(yapp==1,2),'+r','LineWidth',2);
h2=plot(Xapp(yapp==-1,1),Xapp(yapp==-1,2),'db','LineWidth',2);
xsup = Xapp(pos,:);
h3=plot(xsup(:,1),xsup(:,2),'ok','LineWidth',2);
axis([-3 3 -3 3]);

