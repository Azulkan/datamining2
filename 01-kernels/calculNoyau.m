function K = calculNoyau(xi, b)
%CALCULNOYAU Summary of this function goes here
%   Detailed explanation goes here
    for i=1:length(xi)
        d = ((xi(i,1) - xi(:,1)) .^ 2 + (xi(i,2) - xi(:,2)) .^ 2);
        K(:,i) = exp(-d/b);
    end  
end

