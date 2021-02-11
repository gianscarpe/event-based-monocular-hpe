
% add path and load data
shape_root = './shapeconvex/release/';
addpath([shape_root 'ssr']);
addpath([shape_root 'utils']);
shape_data = load('shapeconvex/shapeDict_h36m.mat');

exp_name = 'hg-256-pred';

split = 'val';

save_root = ['./shapeconvex/res_' exp_name '/h36m_' split '/'];
makedir(save_root);

% set opt and init dataset
opt.data = './data/h36m/';
opt.inputRes = 64;
opt.inputResHG = 256;
opt.hg = true;
opt.penn = true;
dataset = h36m(opt, split);

% start parpool
if ~exist('poolsize','var')
    poolobj = parpool();
else
    poolobj = parpool(poolsize);
end

% reading annotations
fprintf('processing shapeconvex on h36m ... \n');
parfor i = 1:dataset.size()
    % skip if vis file exists
    save_file = [save_root sprintf('%05d.mat',i)];
    if exist(save_file,'file')
        continue
    end
    fprintf('%05d/%05d  ',i,dataset.size());
    tt = tic;
    % load 2D prediction
    pred_file = sprintf('./exp/h36m/%s/eval_%s/%05d.mat',exp_name,split,i);
    pred = load(pred_file);
    pred = pred.eval;
    % convert to h36m format
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    pred_ = zeros(17,2);
    pred_(joints,:) = pred;
    pred_(1,:) = (pred(8,:) + pred(9,:))/2;
    pred_(8,:) = (pred(2,:) + pred(3,:) + pred(8,:) + pred(9,:))/4;
    pred_(9,:) = (pred(1,:) + pred(2,:) + pred(3,:))/3;
    pred_(11,:) = pred(1,:);
    pred = pred_;
    X = pred(:,1)';
    Y = pred(:,2)';
    % compute 3d points
    W = normalizeS([X; Y]);
    S = ssr2D3D_wrapper(W,shape_data.B,'convex');
    % convert to single and save
    S = single(S);
    % save to file
    shapeconvex_save(save_file, S);
    time = toc(tt);
    fprintf('tot: %8.3f sec.  \n',time);
end
fprintf('done.\n');

% delete parpool
delete(poolobj);