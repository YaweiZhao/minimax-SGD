function [f,g ] = gradient_check_det(vec_X,n)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
X = reshape(vec_X,n,n);
f = -1/2*logdet(X*X');
g = -1*vec(inv(X'));
end

