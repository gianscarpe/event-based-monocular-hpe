
exp_name = 'res-64';

% split = 'train';
split = 'val';

% set vis root
vis_root = ['./output/vis_' exp_name '/penn_' split '/'];

% load annotations
ind2sub = hdf5read(['./data/penn-crop/' split '.h5'],'ind2sub');
part = hdf5read(['./data/penn-crop/' split '.h5'],'part');
ind2sub = permute(ind2sub,[2 1]);
part = permute(part,[3 2 1]);

% load predictions
preds = load(['./exp/penn-crop/' exp_name '/preds_' split '.mat']);
joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
preds_ = zeros(size(preds.repos,1),17,3);
preds_(:,joints,:) = preds.repos;
preds_(:,1,:) = (preds.repos(:,8,:) + preds.repos(:,9,:))/2;
preds_(:,8,:) = (preds.repos(:,2,:) + preds.repos(:,3,:) + preds.repos(:,8,:) + preds.repos(:,9,:))/4;
preds_(:,9,:) = (preds.repos(:,1,:) + preds.repos(:,2,:) + preds.repos(:,3,:))/3;
preds_(:,11,:) = preds.repos(:,1,:);
preds = permute(preds_,[1,3,2]);
assert(size(preds,1) == size(ind2sub,1));

% load posSkel
db = H36MDataBase.instance();
posSkel = db.getPosSkel();
Features{1} = H36MPose3DPositionsFeature();
[~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');

% set opt and init dataset
opt.data = './data/penn-crop';
opt.inputRes = 64;
opt.inputResHG = 256;
opt.hg = false;
dataset = penn_crop(opt, split);
opt.hg = true;
dataset_hg = penn_crop(opt, split);

% set vis params
pa = [0 1 1 2 3 4 5 2 3 8 9 10 11];
p_no = numel(pa);
msize = 4;
partcolor = {'b','b','b','g','r','g','r','g','r','g','r','g','r'};

% visualize first min(K,len) videos for each action
K = 3;
list_seq = dir('./data/penn-crop/labels/*.mat');
list_seq = {list_seq.name}';
num_seq = numel(list_seq);
action = cell(num_seq,1);
for i = 1:num_seq
    lb_file = ['./data/penn-crop/labels/' list_seq{i}];
    anno = load(lb_file);
    assert(ischar(anno.action));
    action{i} = anno.action;
end
[list_act,~,ia] = unique(action, 'stable');
seq = unique(ind2sub(:,1));
keep = false(numel(seq),1);
for i = 1:numel(list_act)
    ii = find(ismember(seq,find(ia == i)));
    keep(ii(1:min(numel(ii),K))) = true;
end
seq = seq(keep);
run = find(ismember(ind2sub(:,1),seq))';

% init figure
figure(1);
set(gcf,'Position',[2 26 1700 440]);
clear hi hh hs1 hs2

% load libraries
libimg = img();

fprintf('visualizing penn predictions ... \n');
for i = run
    tic_print(sprintf('%05d/%05d\n',find(i == run),numel(run)));
    sid = ind2sub(i,1);
    fid = ind2sub(i,2);    
    
    vis_dir = [vis_root num2str(sid,'%04d') '/'];
    vis_file = [vis_dir num2str(fid,'%06d') '.png'];
    if exist(vis_file,'file')
        continue
    end
    makedir(vis_dir);
    
    % show image
    im_file = ['data/penn-crop/frames/' num2str(sid,'%04d') '/' num2str(fid,'%06d') '.jpg'];
    im = imread(im_file);
    if exist('hi','var')
        delete(hi);
    end
    hi = subplot('Position',[0.03+0/4 0.05 1/4-0.06 0.9]);
    imshow(im); hold on;
    
    % draw annotation
    for child = 2:p_no
        x1 = part(i,pa(child),1);
        y1 = part(i,pa(child),2);
        x2 = part(i,child,1);
        y2 = part(i,child,2);
        % skip invisible joints
        if x2 ~= 0 && y2 ~= 0
            plot(x2, y2, 'o', ...
                'color', partcolor{child}, ...
                'MarkerSize', msize, ...
                'MarkerFaceColor', partcolor{child});
            if x1 ~= 0 && y1 ~= 0
                plot(x1, y1, 'o', ...
                    'color', partcolor{child}, ...
                    'MarkerSize', msize, ...
                    'MarkerFaceColor', partcolor{child});
                line([x1 x2], [y1 y2], ...
                    'color', partcolor{child}, ...
                    'linewidth',round(msize/2));
            end
        end
    end
    
    % draw heatmap
    [im, ~, ~, ~] = dataset_hg.get(i);
    im = permute(im, [2 3 1]);
    [hm, ~, ~, ~] = dataset.get(i);
    if exist('hh','var')
        delete(hh);
    end
    hh = subplot('Position',[0.03+1/4 0.05 1/4-0.06 0.9]);
    inp64 = imresize(double(im),[64 64]) * 0.3;
    colorHms = cell(size(hm,1),1);
    for j = 1:size(hm,1)
        colorHms{j} = libimg.colorHM(squeeze(hm(j,:,:)));
        colorHms{j} = colorHms{j} * 255 * 0.7 + permute(inp64,[3 1 2]);
    end
    totalHm = libimg.compileImages(colorHms, 4, 4, 64);
    totalHm = permute(totalHm,[2 3 1]);
    totalHm = uint8(totalHm);
    imshow(totalHm);

    % show 3D skeleton (view 1)
    if exist('hs1','var')
        delete(hs1);
    end
    hs1 = subplot('Position',[0.03+2/4 0.05 1/4-0.06 0.9]);
    set(gca,'fontsize',6);
    pred = permute(preds(i,:,:),[2 3 1]);
    V = pred;
    V([2 3],:) = V([3 2],:);
    showPose(V,posSkel);
    minx = -1000; maxx = 1000;
    miny = -1000; maxy = 1000;
    minz = -1000; maxz = 1000;
    axis([minx maxx miny maxy minz maxz]);
    set(gca,'ZDir','reverse');
    view([6,10]);
    
    % show 3D skeleton (view 2)
    if exist('hs2','var')
        delete(hs2);
    end
    hs2 = subplot('Position',[0.03+3/4 0.05 1/4-0.06 0.9]);
    set(gca,'fontsize',6);
    pred = permute(preds(i,:,:),[2 3 1]);
    V = pred;
    V([2 3],:) = V([3 2],:);
    showPose(V,posSkel);
    minx = -1000; maxx = 1000;
    miny = -1000; maxy = 1000;
    minz = -1000; maxz = 1000;
    axis([minx maxx miny maxy minz maxz]);
    set(gca,'ZDir','reverse');
    view([85,10]);

    set(gcf,'PaperPositionMode','auto');
    print(gcf,vis_file,'-dpng','-r0');
end
fprintf('done.\n');

close;