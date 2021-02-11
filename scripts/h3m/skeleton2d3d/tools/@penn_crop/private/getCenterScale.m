function [ center, scale ] = getCenterScale( obj, im )

assert(ndims(im) == 3);
w = size(im,3);
h = size(im,2);
x = (w+1)/2;
y = (h+1)/2;
scale = max(w,h)/200;
% Small adjustment so cropping is less likely to take feet out
y = y + scale * 15;
scale = scale * 1.25;
center = [x, y]; 

end

