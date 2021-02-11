
% add path and load data
shape_root = './shapeconvex/release/';
addpath([shape_root 'utils']);
shape_data = load('shapeconvex/shapeDict_h36m.mat');

exp_name = 'hg-256-pred';

split = 'val';

interval = 101;

% set body joint config
pa = [0 1 2 3 1 5 6 1 8 8 10 11 8 13 14];
p_no = numel(pa);

% set vis params
msize = 4;
partcolor = {'b','b','r','r','b','g','g','b','b','b','r','r','b','g','g'};

% set directories
pose_root = ['./shapeconvex/res_' exp_name '/h36m_' split '/'];
save_root = ['./shapeconvex/vis_' exp_name '/h36m_' split '/'];
makedir(save_root);

% init dataset
dataset = load(['./data/h36m/' split '.mat']);

figure(1);
set(gcf,'Position',[2 26 640 320]);

% reading annotations
fprintf('visualizing shapeconvex on h36m ... \n');
for i = 1:interval:size(dataset.ind2sub,1)
    tic_print(sprintf('%05d/%05d\n',i,size(dataset.ind2sub,1)));
    sid = dataset.ind2sub(i,1);
    aid = dataset.ind2sub(i,2);
    bid = dataset.ind2sub(i,3);
    fid = dataset.ind2sub(i,4);
    cam = mod(i-1,4)+1;

    % skip if vis file exists
    save_file = sprintf('%s/%02d_%02d_%1d_%04d_%1d.png',save_root,sid,aid,bid,fid,cam);
    if exist(save_file,'file')
        continue
    end

    % load 2D prediction
    pred_file = sprintf('./exp/h36m/%s/eval_%s/%05d.mat',exp_name,split,i);
    pred = load(pred_file);
    pred = pred.eval;
    % convert from 13 to 15 joints
    X = convert_joint(pred(:,1)');
    Y = convert_joint(pred(:,2)');
    % convert joint order
    X = X(:,[15,9,11,13,8,10,12,14,1,3,5,7,2,4,6]);
    Y = Y(:,[15,9,11,13,8,10,12,14,1,3,5,7,2,4,6]);
    % load predicted 3d pose
    pose_file = [pose_root sprintf('%05d.mat',i)];
    pose = load(pose_file);
    S = pose.S;

    % plot image and annotation
    if exist('h1','var')
        delete(h1);
    end
    h1 = subplot('Position',[0.00+0/2 0.00 1/2-0.00 1.00]);
    % read image
    im_file = sprintf('./data/h36m/frames/%02d/%02d/%1d/%1d_%04d.jpg',sid,aid,bid,cam,fid);
    im = imread(im_file);
    imshow(im); hold on;
    % setup_im_gcf(size(im,1),size(im,2));
    for child = 2:p_no
        x1 = X(1,pa(child));
        y1 = Y(1,pa(child));
        x2 = X(1,child);
        y2 = Y(1,child);
        plot(x2, y2, 'o', ...
            'color', partcolor{child}, ...
            'MarkerSize', msize, ...
            'MarkerFaceColor', partcolor{child});
        plot(x1, y1, 'o', ...
            'color', partcolor{child}, ...
            'MarkerSize', msize, ...
            'MarkerFaceColor', partcolor{child});
        line([x1 x2], [y1 y2], ...
            'color', partcolor{child}, ...
            'linewidth',round(msize/2));
    end
    
    % plot 3d points
    if exist('h2','var')
        delete(h2);
    end
    h2 = subplot('Position',[0.00+1/2 0.00 1/2-0.00 1.00]);
    vis3Dskel(S,shape_data.skel, ...
        'viewpoint',[30 30], ...
        'showcam','true', ...
        'mode','stick');
    axis on;
    grid on;

    % save vis to file
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpng','-r0');
end
fprintf('\n');

close;
