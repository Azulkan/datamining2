function [ alpha ] = learn( K,ya )
%LEARN Summary of this function goes here
%   Detailed explanation goes here
    I = eye(length(ya));
    lam = 10e-6;
    alpha = (K + lam * I) \ ya;
end

