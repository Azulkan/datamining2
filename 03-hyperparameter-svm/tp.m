%% TP 3 - SVM et Réglage des hyper-paramètres


%% 1. Lecture des données
[Xi ,yi] = read_libsvm('splice.a');
[na ,p] = size(Xi);
[Xt ,yt] = read_libsvm('splice.t');
[nt , p] = size(Xt);

%% 3. Grille logarithmique
b_grid = logspace (0.5 ,2 ,25);
C_grid = logspace (-0.5,2,25);

%% 5
% a)
percent = 0.1;
[Xa ,ya ,Xval ,yval] = split_data(Xi,yi,percent);
la = eps ^.5;
kernel='gaussian';
[b_opt ,C_opt ,Err1] = svm_CV(Xa,ya,Xval ,yval ,b_grid ,C_grid ,kernel ,la);
[Xa ,ya ,Xval ,yval] = split_data(Xi,yi,percent);
[b_opt ,C_opt ,Err2] = svm_CV(Xa,ya,Xval ,yval ,b_grid ,C_grid ,kernel ,la);
Err = (Err1+ Err2)/2;
[Errmin  C_ind] = min(min(Err));
C_opt = C_grid(C_ind);
[Errmin  b_ind] = min(min(Err'));
b_opt = b_grid(b_ind);
%%
% b)
contour(log(b_grid), log(C_grid), Err ,[10:.5:50]);

%% 6
% a)
b_min = b_grid(b_ind -2);
b_max =   b_grid(b_ind +2);
b_grid = linspace(b_min ,b_max ,10);
C_min =   C_grid(C_ind -2);
C_max =   C_grid(C_ind +2);
C_grid = linspace(C_min ,C_max ,10);