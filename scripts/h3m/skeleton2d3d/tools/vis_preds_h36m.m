
exp_name = 'res-64';

% split = 'train';
split = 'val';

% set vis root
vis_root = ['./output/vis_' exp_name '/h36m_' split '/'];
makedir(vis_root);

% load annotations
anno = load(['./data/h36m/' split '.mat']);
ind2sub = anno.ind2sub;

% load predictions
preds = load(['./exp/h36m/' exp_name '/preds_' split '.mat']);
joints = [10,15,12,16,13,17,14,2,5,3,6,4,7];
repos = zeros(size(preds.repos,1),17,3);
repos(:,joints,:) = preds.repos;
repos(:,1,:) = (preds.repos(:,8,:) + preds.repos(:,9,:))/2;
repos(:,8,:) = (preds.repos(:,2,:) + preds.repos(:,3,:) + preds.repos(:,8,:) + preds.repos(:,9,:))/4;
repos(:,9,:) = (preds.repos(:,1,:) + preds.repos(:,2,:) + preds.repos(:,3,:))/3;
repos(:,11,:) = preds.repos(:,1,:);
repos = permute(repos,[1,3,2]);
assert(size(repos,1) == size(ind2sub,1));
trans = preds.trans;
focal = preds.focal;

% load groud truth
poses = zeros(size(preds.poses,1),17,3);
poses(:,joints,:) = preds.poses;
poses(:,1,:) = (preds.poses(:,8,:) + preds.poses(:,9,:))/2;
poses(:,8,:) = (preds.poses(:,2,:) + preds.poses(:,3,:) + preds.poses(:,8,:) + preds.poses(:,9,:))/4;
poses(:,9,:) = (preds.poses(:,1,:) + preds.poses(:,2,:) + preds.poses(:,3,:))/3;
poses(:,11,:) = preds.poses(:,1,:);
poses = permute(poses,[1,3,2]);
assert(size(poses,1) == size(ind2sub,1));

% load posSkel
db = H36MDataBase.instance();
posSkel = db.getPosSkel();
Features{1} = H36MPose3DPositionsFeature();
[~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');

% init camera
CameraVertex = zeros(5,3);
CameraVertex(1,:) = [0 0 0];
CameraVertex(2,:) = [-250  250  500];
CameraVertex(3,:) = [ 250  250  500];
CameraVertex(4,:) = [-250 -250  500];
CameraVertex(5,:) = [ 250 -250  500];
IndSetCamera = {[1 2 3 1] [1 4 2 1] [1 5 4 1] [1 5 3 1] [2 3 4 5 2]};

% set color for gt
clr = {'k','c','c','k','m','m','k','k','k','k','k','m','m','k','c','c'};

% sample frames
run = 1:101:size(ind2sub,1);

% init figure
figure(1);
set(gcf,'Position',[2 26 1376 750]);
clear hr hs

fprintf('visualizing h36m predictions ... \n');
for i = run
    tic_print(sprintf('%05d/%05d\n',find(i == run),numel(run)));
    sid = ind2sub(i,1);
    aid = ind2sub(i,2);
    bid = ind2sub(i,3);
    fid = ind2sub(i,4);
    vis_file = [vis_root sprintf('%02d_%02d_%1d_%04d.png',sid,aid,bid,fid)];
    if exist(vis_file,'file')
        continue
    end
    
    % initialize sub-figure: 3D skeleton relative to center
    if exist('hr','var')
        delete(hr);
    end
    hr = subplot('Position',[0.03+0/2 0.075 1/2-0.06 0.85]);
    set(gca,'fontsize',6);
    % draw ground-truth skeleton
    pose = permute(poses(i,:,:),[2 3 1]);
    pose = pose - repmat(mean(pose,2),[1 size(pose,2)]);
    V = pose;
    V([2 3],:) = V([3 2],:);
    showPose(V,posSkel);
    % draw predicted pose
    pred = permute(repos(i,:,:),[2 3 1]);
    V = pred;
    V([2 3],:) = V([3 2],:);
    hpos = showPose(V,posSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'color',clr{k});
    end
    % set figure
    minx = -1000; maxx = 1000;
    miny = -1000; maxy = 1000;
    minz = -1000; maxz = 1000;
    axis([minx maxx miny maxy minz maxz]);
    set(gca,'ZDir','reverse');
    view([35,30]);
    ht = title('relative position to center');
    set(ht,'fontsize',16);
    
    % initialize sub-figure: 3D skeleton in camera coordinates
    if exist('hs','var')
        delete(hs);
    end
    hs = subplot('Position',[0.03+1/2 0.075 1/2-0.06 0.85]);
    set(gca,'fontsize',6);
    % draw ground-truth skeleton
    pose = permute(poses(i,:,:),[2 3 1]);
    V = pose;
    V([2 3],:) = V([3 2],:);
    showPose(V,posSkel);
    % draw predicted pose
    pred = permute(repos(i,:,:),[2 3 1]);
    pred = pred + repmat(permute(trans(i,:),[2 1]),[1 size(pred,2)]);
    V = pred;
    V([2 3],:) = V([3 2],:);
    hpos = showPose(V,posSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',2);
        set(hpos(k+1),'color',clr{k});
    end
    % draw camera
    CVWorld = CameraVertex;
    CVWorld(:,[2 3]) = CVWorld(:,[3 2]);
    hc = zeros(size(CameraVertex,1),1);
    for ind = 1:length(IndSetCamera)
        hc(ind) = patch( ...
            CVWorld(IndSetCamera{ind},1), ...
            CVWorld(IndSetCamera{ind},2), ...
            CVWorld(IndSetCamera{ind},3), ...
            [0.5 0.5 0.5]);
    end
    % set figure
    minx = -2000; maxx = 2000;
    miny =     0; maxy = 5000;
    minz = -2000; maxz = 2000;
    axis([minx maxx miny maxy minz maxz]);
    set(gca,'ZDir','reverse');
    view([35,30]);
    ht = title('camera coordiantes');
    set(ht,'fontsize',16);
    
    % save figure
    set(gcf,'PaperPositionMode','auto');
    print(gcf,vis_file,'-dpng','-r0');
end

fprintf('done.\n');

close;