function [ cam, pind, pose, proj, depth, trial ] = sample_pose_proj( pose_cand )
% input
%   pose_cand:  N x 51
% output
%   pose:       3 x 17
%   proj:       2 x 17
%   depth:      1 x 17

% sample pose
pind = randi(size(pose_cand,1));
pose = pose_cand(pind,:);
pose = reshape(pose,[3 numel(pose)/3])';

% normalize to zero mean
cent = mean(pose,1);
pose = pose - repmat(cent,[size(pose,1) 1]);

% set fixed camera params
cam.res = [64 64];
cam.c = round(cam.res / 2);

trial = 0;

while true
    trial = trial + 1;
    
    % sample focal length
    mu = 1150 * 0.064;
    std = 450 * 0.064;
    cam.f = repmat(randn() * std + mu, [1 2]);
    
    % sample translation
    r = rand() * 4000 + 1000;                 % 1000 to 5000
    az = (rand() - 0.5000) * 360 * pi / 180;  % -180 to +180
    el = (rand() - 0.1429) *  35 * pi / 180;  %   -5 to  +30
    x = r * cos(el) * cos(az);
    y = r * cos(el) * sin(az);
    z = r * sin(el);
    cam.T = [x y z];
    
    % sample rotation
    r1 = (rand() - 0.5000) *  10 * pi / 180;  %  -5  to   +5
    r2 = (rand() - 0.5000) *  90 * pi / 180;  % -45  to  +45
    r3 = (rand() - 0.8571) *  35 * pi / 180;  % -30  to   +5
    cam.R = angle2dcm(r1, r2, r3,'ZYX') * ...
        angle2dcm(pi/2, 0, -pi/2,'ZYX') * ...
        angle2dcm(az, -el, 0,'ZYX');
    
    % get projection and depth
    [proj, depth] = cam_project(pose, cam.R, cam.T, cam.f, cam.c);
    
    % skip if any point is behind the image plane
    if any(depth < cam.f(1))
        continue
    end
    
    % skip if number of visible joints is below threshold
    joint_thres = 0.999;
    c1 = proj(:,1) >= 1 & proj(:,1) <= cam.res(1);
    c2 = proj(:,2) >= 1 & proj(:,2) <= cam.res(2);
    if sum(c1 & c2) < size(proj,1) * joint_thres
        continue
    end
    
    % skip if human is too small
    area_thres = 0.1;
    x1 = min(proj(:,1));
    y1 = min(proj(:,2));
    x2 = max(proj(:,1));
    y2 = max(proj(:,2));
    area = (x2 - x1) * (y2 - y1);
    if area < prod(cam.res) * area_thres
        continue
    end
    
    break
end

pose = pose';
proj = proj';
depth = depth';

end