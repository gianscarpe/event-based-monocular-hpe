% load gt and pred
%   dataset:       dataset
%   repos:         gt
%   preds_convex:  pred for shapeconvex
%   preds_skel3d:  pred for 3d skeleton converter

% init dataset and convert to penn format
dataset = load('./data/h36m/val.mat');
joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
ind2 = [];
for i = joints
    for j = 1:2
        ind2 = [ind2; (i-1)*2+j];  %#ok
    end
end
ind3 = [];
for i = joints
    for j = 1:3
        ind3 = [ind3; (i-1)*3+j];  %#ok
    end
end
dataset.coord_w = dataset.coord_w(:,ind3);
dataset.coord_c = dataset.coord_c(:,:,ind3);
dataset.coord_p = dataset.coord_p(:,:,ind2);

% load gt
fprintf('loading groundtruths from h36m ... \n');
repos = zeros(size(dataset.ind2sub,1),size(dataset.coord_c,3)/3,3);
for i = 1:size(dataset.ind2sub,1)
    cam = mod(i-1,4) + 1;
    pose = reshape(dataset.coord_c(cam,i,:),[3 size(dataset.coord_c,3)/3])';
    pose = double(pose);
    cntr = mean(pose,1);
    repos(i,:,:) = pose - repmat(cntr,[size(pose,1) 1]);
end
fprintf('done.\n');

% load shapeconvex pred
preds_root = './shapeconvex/res_hg-256-pred/h36m_val/';
if exist(preds_root,'dir')
    % get mean limb length
    conn = [ 2  1; 3  1; 4  2; 5  3; 6  4; 7  5; 8  2; 9  3;10  8;11  9;12 10;13 11];
    coord_w = reshape(dataset.coord_w,[size(dataset.coord_w,1) 3 13]);
    d1 = coord_w(:,:,conn(:,1));
    d2 = coord_w(:,:,conn(:,2));
    mu = mean(sqrt(sum((d1 - d2).^2,2)),1);
    mu = permute(mu,[3 2 1]);
    % read prediction
    fprintf('loading shapeconvex prediction ... \n');
    preds_convex = zeros(size(dataset.ind2sub,1),size(dataset.coord_c,3)/3,3);
    for i = 1:size(dataset.ind2sub,1)
        tic_print(sprintf('%05d/%05d\n',i,size(dataset.ind2sub,1)));
        % load predicted 3d pose
        pred_file = [preds_root sprintf('%05d.mat',i)];
        pred = load(pred_file);
        S = pred.S;
        % convert to penn format
        joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
        pred = S(:,joints)';
        % scale pred to minimize error with length prior
        p1 = pred(conn(:,1),:);
        p2 = pred(conn(:,2),:);
        len = sqrt(sum((p1 - p2).^2,2));
        c = median(mu ./ len);
        preds_convex(i,:,:) = pred * c;
    end
    fprintf('done.\n');
end

% load 3d skeleton converter pred
preds_file = './exp/h36m/hg-256-res-64-hg1-hgfix/preds_val.mat';
if exist(preds_file,'file')
    fprintf('loading 3d skeleton converter prediction ... \n');
    preds = load(preds_file);
    preds_skel3d = preds.repos;
    fprintf('done.\n');
end