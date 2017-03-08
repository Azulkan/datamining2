function [Xsup, alpha, b] = MySVMClass(Xapp, yapp, C, kernel, kerneloption)
    lambda = 1e-7;
    [Xsup, alpha, b, pos] = svmclass(Xapp, yapp, C, lambda,kernel, kerneloption, 1);
end