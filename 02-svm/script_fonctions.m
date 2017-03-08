%% Same problem with n = 500, bandwidth = 0.05

% SVM non-linéaire
clear; close all;

%% Generate the data 
n = 1000;
[Xapp,yapp,Xtest,ytest] = dataset_KM('checkers', n, n^2);
[n, p] = size(Xapp);

figure(1); hold on
% Les labels 1
%h1=plot(Xapp(yapp==1,1),Xapp(yapp==1,2),'+r','LineWidth',2);
% Les labels -1
%h2=plot(Xapp(yapp==-1,1),Xapp(yapp==-1,2),'db','LineWidth',2);

%% Compute the C-SVM with gaussian kernel (kerneloption = 0.05)
D = (Xapp*Xapp');
N = diag(D);
D = -2*D + N*ones(1,n) + ones(n,1)*N';
kerneloption = .05;


%% Same, but with SVM kernel function from the toolbox
kernel = 'gaussian';
K =svmkernel(Xapp,kernel,kerneloption);

C = 10000;


[Xsup,alpha,b] = MySVMClass(Xapp, yapp, C, kernel, kerneloption);
[y_pred] = MySVMVal(Xtest,Xsup,alpha,b,kernel, kerneloption);