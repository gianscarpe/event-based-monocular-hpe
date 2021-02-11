
% load gt and pred
%   dataset:       dataset
%   repos:         gt
%   preds_convex:  pred for shapeconvex
%   preds_skel3d:  pred for 3d skeleton converter
load_res;

% set joint names
jnames = {'head', ...
    'r.sho', 'l.sho', 'r.elb', 'l.elb', 'r.wri', 'l.wri', ...
    'r.hip', 'l.hip', 'r.kne', 'l.kne', 'r.ank', 'l.ank'};

% print mpmje for shapeconvex
fprintf('\n');
fprintf('mpmje: Zhou et al.\n');
if exist('preds_convex','var')
    mpmje = mean(sqrt(sum((repos - preds_convex).^2,3)),1);
    for i = 1:numel(mpmje)
        fprintf('%-5s  %5.1f\n',jnames{i},mpmje(i));
    end
    fprintf('%-5s  %5.1f\n','avg',mean(mpmje));
else
    fprintf('no result found.\n');
end

% print mpmje for 3d skeleton converter
fprintf('\n');
fprintf('mpmje: ours\n');
if exist('preds_skel3d','var')
    mpmje = mean(sqrt(sum((repos - preds_skel3d).^2,3)),1);
    for i = 1:numel(mpmje)
        fprintf('%-5s  %5.1f\n',jnames{i},mpmje(i));
    end
    fprintf('%-5s  %5.1f\n','avg',mean(mpmje));
else
    fprintf('no result found.\n');
end
