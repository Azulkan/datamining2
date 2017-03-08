function y = predRationnel(x, xa, ya )
% Fonction rationnelle (non linéaire)
    alpha = ya;
    b = 0.0001;
    d = ((x(1)-xa(:,1)).^2+(x(2)-xa(:,2)).^2);
    K = 1-(d.^2./(d.^2+b));
    y = alpha'*K;
end