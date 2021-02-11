function [ input, proj, center, scale ] = get( obj, idx )
% This does not produce the exact output as the get() in penn-crop.lua
%   1. input is of type uint8 with values in range [0 255]

% Load image
im = loadImage(obj, idx);

% Get center and scale
[center, scale] = getCenterScale(obj, im);

% Transform image
im = obj.img.crop(im, center, scale, 0, obj.inputRes);

% Get projection
pts = permute(obj.part(idx,:,:),[2 3 1]);
proj = zeros(size(pts));
for i = 1:size(pts,1)
    if pts(i,1) ~= 0 && pts(i,2) ~= 0
        proj(i,:) = obj.img.transform(pts(i,:), center, scale, 0, obj.outputRes, false, false);
    end
end

% Generate heatmap
hm = zeros(obj.outputRes,obj.outputRes,size(pts,1));
for i = 1:size(pts,1)
    if pts(i,1) ~= 0 && pts(i,2) ~= 0
        hm(:,:,i) = obj.img.drawGaussian(hm(:,:,i),round(proj(i,:)),2);
    end
end
hm = permute(hm,[3 1 2]);

% Set input
if obj.hg
    input = im;
else
    input = hm;
end

end

