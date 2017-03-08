function ypred = MySVMVal(Xtest, Xsup, alpha, b, kernel, kerneloption)
    ypred = svmval(Xtest, Xsup, alpha, b, kernel, kerneloption);
end