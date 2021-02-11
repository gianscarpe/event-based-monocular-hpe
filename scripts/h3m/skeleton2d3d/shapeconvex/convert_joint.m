function [ P_new ] = convert_joint( P )
% convert from 13 keypoints to 15 keypoints
% 14: thorax, interpolated from rshou (2) and lshou (3)
% 15: pelvis, interpolated from rhip (8) and lhip (9)
%
% input should be a matrix of size N x 13; could be X, Y or mixture of both

assert(size(P,2) == 13);

P_new = zeros(size(P,1),15);
P_new(:,1:13) = P;
P_new(:,14) = (P(:,2)+P(:,3))/2;
P_new(:,15) = (P(:,8)+P(:,9))/2;

end