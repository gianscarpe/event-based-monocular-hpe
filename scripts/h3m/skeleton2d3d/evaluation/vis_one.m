
makedir(save_root);

split = 'val';

% set opt and init dataset
opt.data = './data/h36m/';
opt.inputRes = 64;
opt.inputResHG = 256;
opt.hg = true;
opt.penn = true;
dataset_mat = h36m(opt, split);

% load posSkel
db = H36MDataBase.instance();
posSkel = db.getPosSkel();
pos2dSkel = posSkel;
for i = 1 :length(pos2dSkel.tree)
    pos2dSkel.tree(i).posInd = [(i-1)*2+1 i*2];
end
Features{1} = H36MPose3DPositionsFeature();
[~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');
[~, pos2dSkel] = Features{1}.select(zeros(0,64), pos2dSkel, 'body');

% set color for gt
clr = {'k','c','c','k','m','m','k','k','k','k','k','m','m','k','c','c'};

% sample frames
run = 1:101:size(dataset.ind2sub,1);

% load libraries
libimg = img();

% init figure
figure(1);
set(gcf,'Position',[2 26 1830 330]);
clear hi hh hs1 hs2 hr1 hr2

fprintf('visualizing prediction ... \n');
for i = run
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
    pred_file = sprintf('%s/%05d.mat',eval_dir,i);
    pred = load(pred_file);
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    pred_ = zeros(17,2);
    pred_(joints,:) = pred.eval;
    pred_(1,:) = (pred.eval(8,:) + pred.eval(9,:))/2;
    pred_(8,:) = (pred.eval(2,:) + pred.eval(3,:) + pred.eval(8,:) + pred.eval(9,:))/4;
    pred_(9,:) = (pred.eval(1,:) + pred.eval(2,:) + pred.eval(3,:))/3;
    pred_(11,:) = pred.eval(1,:);
    pred = pred_;
    % draw 2D prediction
    if exist('hp','var')
        delete(hp);
    end
    hp = subplot('Position',[0.00+0/6 0.00 1/6-0.00 1.00]);
    im_file = sprintf('./data/h36m/frames/%02d/%02d/%1d/%1d_%04d.jpg',sid,aid,bid,cam,fid);
    im = imread(im_file);
    imshow(im); hold on;
    show2DPose(permute(pred,[2 1]),pos2dSkel);
    axis off;
    
    % load heatmap
    hmap_file = sprintf('%s/%05d.mat',hmap_dir,i);
    hmap = load(hmap_file);
    hmap = hmap.hmap;
    % draw heatmap
    if exist('hh','var')
        delete(hh);
    end
    hh = subplot('Position',[0.00+1/6 0.00 1/6-0.00 1.00]);
    [input, ~, ~, ~, ~] = dataset_mat.get(i);
    input = permute(input, [2 3 1]);
    inp64 = imresize(double(input),[64 64]) * 0.3;
    colorHms = cell(size(hmap,1),1);
    for j = 1:size(hmap,1)
        colorHms{j} = libimg.colorHM(squeeze(hmap(j,:,:)));
        colorHms{j} = colorHms{j} * 255 * 0.7 + permute(inp64,[3 1 2]);
    end
    totalHm = libimg.compileImages(colorHms, 4, 4, 64);
    totalHm = permute(totalHm,[2 3 1]);
    totalHm = uint8(totalHm);
    imshow(totalHm);

    % convert to h36m format
    joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
    repos_i = zeros(17,3);
    repos_i(joints,:) = repos(i,:,:);
    repos_i(1,:) = (repos(i,8,:) + repos(i,9,:))/2;
    repos_i(8,:) = (repos(i,2,:) + repos(i,3,:) + repos(i,8,:) + repos(i,9,:))/4;
    repos_i(9,:) = (repos(i,1,:) + repos(i,2,:) + repos(i,3,:))/3;
    repos_i(11,:) = repos(i,1,:);
    pred = zeros(17,3);
    pred(joints,:) = preds(i,:,:);
    pred(1,:) = (preds(i,8,:) + preds(i,9,:))/2;
    pred(8,:) = (preds(i,2,:) + preds(i,3,:) + preds(i,8,:) + preds(i,9,:))/4;
    pred(9,:) = (preds(i,1,:) + preds(i,2,:) + preds(i,3,:))/3;
    pred(11,:) = preds(i,1,:);
    % show 3D skeleton
    for j = 1:4
        if j == 1
            if exist('hs1','var')
                delete(hs1);
            end
            hs1 = subplot('Position',[0.03+2/6 0.07 1/6-0.04 0.93]);
        end
        if j == 2
            if exist('hs2','var')
                delete(hs2);
            end
            hs2 = subplot('Position',[0.02+3/6 0.07 1/6-0.03 0.93]);
        end
        if j == 3
            if exist('hr1','var')
                delete(hr1);
            end
            hr1 = subplot('Position',[0.02+4/6 0.07 1/6-0.035 0.93]);
        end
        if j == 4
            if exist('hr2','var')
                delete(hr2);
            end
            hr2 = subplot('Position',[0.02+5/6 0.07 1/6-0.035 0.93]);
        end
        set(gca,'fontsize',6);
        V = pred';
        if j == 3 || j == 4
            V(1,:) = V(1,:) - 500;
            V(3,:) = V(3,:) - 500;
        end
        V([2 3],:) = V([3 2],:);
        hpos = showPose(V,posSkel);
        for k = 1:numel(hpos)-1
            set(hpos(k+1),'linewidth',3);
        end
        V = repos_i';
        if j == 3 || j == 4
            V(1,:) = V(1,:) + 500;
            V(3,:) = V(3,:) + 500;
        end
        V([2 3],:) = V([3 2],:);
        hpos = showPose(V,posSkel);
        for k = 1:numel(hpos)-1
            set(hpos(k+1),'linewidth',3);
            set(hpos(k+1),'color',clr{k});
        end
        minx = -1000; maxx = 1000;
        miny = -1000; maxy = 1000;
        minz = -1000; maxz = 1000;
        axis([minx maxx miny maxy minz maxz]);
        set(gca,'ZTick',-1000:200:1000);
        set(gca,'ZDir','reverse');
        if j == 1 || j == 3
            view([6,10]);
        end
        if j == 2 || j == 4
            view([85,10]);
        end
    end
    hs1 = subplot('Position',[0.03+2/6 0.07 1/6-0.04 0.93]);
    e = sqrt(sum((repos_i - pred).^2,2));
    e = mean(e);
    t = title(['error: ' num2str(e,'%.2f')]);
    set(t,'FontSize',10);
    
    % save vis to file
    set(gcf,'PaperPositionMode','auto');
    print(gcf,save_file,'-dpng','-r0');
end
fprintf('done.\n');

close;