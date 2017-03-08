function y = pred(x, xa, ya )
% Calcule la somme des potentiels
% Fonction exponentielle (non lineaire) 
    alpha = ya;
    b = 0.01;
    d = ((x(1) - xa(:,1)) .^ 2 + (x(2) - xa(:,2)) .^ 2);
    K = exp(-d/b);
    y = alpha' * K;
end

