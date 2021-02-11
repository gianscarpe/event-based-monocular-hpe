function [ pose, cntr ] = normalizePose( obj, pose )

cntr = mean(pose,1);
pose = pose - repmat(cntr,[size(pose,1) 1]);
cntr = reshape(cntr, [3 1]);

end

