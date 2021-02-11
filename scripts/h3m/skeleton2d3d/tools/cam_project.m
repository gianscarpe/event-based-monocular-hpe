function [ p, D ] = cam_project( P, R, T, f, c )
% input
%   P:  N x 3
%   R:  3 x 3
%   T:  1 x 3
%   f:  1 x 2
%   c:  1 x 2
% output
%   p:  N x 2
%   D:  N x 1

N = size(P,1);
X = R * (P' - T' * ones(1,N));
p = X(1:2,:) ./ ([1; 1] * X(3,:));
p = ones(N,1) * f .* p' + ones(N,1) * c;
D = X(3,:)';

end

