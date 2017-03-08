function [y] = preda(x, xa, b, alpha)
    d = ((x(1) - xa(:,1)) .^ 2 + (x(2) - xa(:,2)) .^ 2);
    K = exp(-d/b);
    y = alpha' * K;
end

