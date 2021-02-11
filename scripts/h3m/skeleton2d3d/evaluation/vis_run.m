
% load gt and pred
%   dataset:       dataset
%   repos:         gt
%   preds_convex:  pred for shapeconvex
%   preds_skel3d:  pred for 3d skeleton converter
load_res;

% visualize for shapeconvex
fprintf('\n');
fprintf('shapeconvex:\n');
if exist('preds_convex','var')
    preds = preds_convex;
    hmap_dir = 'exp/h36m/hg-256-pred/hmap_val';
    eval_dir = 'exp/h36m/hg-256-pred/eval_val';
    save_root = 'evaluation/vis_shapeconvex';
    vis_one;
else
    fprintf('no result found.\n');
end

% visualize for 3d skeleton converter
fprintf('\n');
fprintf('3d skeleton converter:\n');
if exist('preds_skel3d','var')
    preds = preds_skel3d;
    hmap_dir = 'exp/h36m/hg-256-res-64-hg1-hgfix/hmap_val';
    eval_dir = 'exp/h36m/hg-256-res-64-hg1-hgfix/eval_val';
    save_root = 'evaluation/vis_skeleton2d3d';
    vis_one;
else
    fprintf('no result found.\n');
end
