function [ input, repos, trans, focal, proj ] = get( obj, idx )
% This does not produce the exact output as the get() in h36m.lua
%   1. input is of type uint8 with values in range [0 255]

% currently only suppors obj.hg == true and train == false
assert(obj.hg == true);

cam = mod(idx-1, 4) + 1;

im = loadImage(obj, idx, cam);
% Transform image
[center, scale] = getCenterScale(obj, im);
im = obj.img.crop(im, center, scale, 0, obj.inputRes);
% Scale focal length
factor = obj.outputRes / (scale * 200);
focal = mean(loadFocal(obj, cam), 2) * factor;
% Load pose
pose_c = loadPoseCamera(obj, idx, cam);
[repos, trans] = normalizePose(obj, pose_c);
% % Load and transform projection
% proj = loadPoseProject(obj, idx, cam);
% for i = 1:size(proj,1)
%     proj(i,:) = obj.img.transform(proj(i,:), center, scale, 0, obj.outputRes, false, false);
% end
proj = [];

% % Generate heatmap
% hm = zeros(obj.outputRes,obj.outputRes,size(pts,1));
% for i = 1:size(proj,1)
%     hm(:,:,i) = obj.img.drawGaussian(hm(:,:,i),round(proj(i,:)),2);
% end
% hm = permute(hm,[3 1 2]);

% Set input
if obj.hg
    input = im;
else
    % input = hm;
end

end

