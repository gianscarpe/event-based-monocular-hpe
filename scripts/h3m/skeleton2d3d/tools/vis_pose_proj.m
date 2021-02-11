
rseed;

vis_dir = 'output/vis_pose_proj/';
makedir(vis_dir);

db = H36MDataBase.instance();

posSkel = db.getPosSkel();
pos2dSkel = posSkel;
for i = 1 :length(pos2dSkel.tree)
    pos2dSkel.tree(i).posInd = [(i-1)*2+1 i*2];
end

data_file = './data/h36m/train.mat';
data = load(data_file);

Features{1} = H36MPose3DPositionsFeature();
[~, posSkel] = Features{1}.select(zeros(0,96), posSkel, 'body');
[~, pos2dSkel] = Features{1}.select(zeros(0,64), pos2dSkel, 'body');

CameraVertex = zeros(5,3);
CameraVertex(1,:) = [0 0 0];
CameraVertex(2,:) = [-250  250  500];
CameraVertex(3,:) = [ 250  250  500];
CameraVertex(4,:) = [-250 -250  500];
CameraVertex(5,:) = [ 250 -250  500];
IndSetCamera = {[1 2 3 1] [1 4 2 1] [1 5 4 1] [1 5 3 1] [2 3 4 5 2]};

libimg = img();

% generate N samples
N = 100;

figure(1);
set(gcf,'Position',[2 26 1575 500]);
clear hs hp hh hd

for n = 1:N
    tic_print(sprintf('%03d/%03d\n',n,N));
    
    % sample pose proj
    [cam, pind, pose, proj, depth, trial] = sample_pose_proj(data.coord_w);
    
    % draw skeleton and camera
    if exist('hs','var')
        delete(hs);
    end
    hs = subplot('Position',[0.035+0/3 0.05 1/3-0.06 0.9]);
    set(gca,'fontsize',6);
    hpos = showPose(pose,posSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',3);
    end
    ylabel('y');
    zlabel('z');
    minx = -3000; maxx = 3000;
    miny = -3000; maxy = 3000;
    minz = -2000; maxz = 2000;
    axis([minx maxx miny maxy minz maxz]);
    view([35,30]);
    CameraVertex(2:end,3) = cam.f(1) / 0.064;
    CVWorld = (cam.R'*CameraVertex')' + repmat(cam.T,[size(CameraVertex,1) 1]);
    hc = zeros(size(CameraVertex,1),1);
    for ind = 1:length(IndSetCamera)
        hc(ind) = patch( ...
            CVWorld(IndSetCamera{ind},1), ...
            CVWorld(IndSetCamera{ind},2), ...
            CVWorld(IndSetCamera{ind},3), ...
            [0.5 0.5 0.5]);
    end
        
    % draw projection
    if exist('hp','var')
        delete(hp);
    end
    hp = subplot('Position',[0.035+1/3 0.05 1/3-0.06 0.9]);
    set(gca,'fontsize',6);
    hpos = show2DPose(proj,pos2dSkel);
    for k = 1:numel(hpos)-1
        set(hpos(k+1),'linewidth',5);
    end
    ylabel('y');
    axis([0 cam.res(1) 0 cam.res(2)]);
    axis ij;
    
    % create heatmaps
    hm = zeros(cam.res(1),cam.res(2),size(proj,2));
    for i = 1:size(proj,2)
        hm(:,:,i) = libimg.drawGaussian(hm(:,:,i),proj(:,i),2);
    end
    
    % save visualized 2d pose
    F = getframe(gca);
    I = frame2im(F);
    
    % draw heamap
    if exist('hh','var')
        delete(hh);
    end
    hh = subplot('Position',[0.035+2/3 0.05 1/3-0.06 0.9]);
    inp64 = imresize(double(I),cam.res) * 0.3;
    colorHms = cell(size(proj,2),1);
    for i = 1:size(proj,2)
        colorHms{i} = libimg.colorHM(hm(:,:,i));
        colorHms{i} = colorHms{i} * 255 * 0.7 + permute(inp64,[3 1 2]);
    end
    totalHm = libimg.compileImages(colorHms, 4, 5, 64);
    totalHm = permute(totalHm,[2 3 1]);
    totalHm = uint8(totalHm);
    imshow(totalHm);
    
    vis_file = [vis_dir sprintf('%05d.png',n)];
    if ~exist(vis_file,'file')
        set(gcf,'PaperPositionMode','auto');
        print(gcf,vis_file,'-dpng','-r0');
    end
end

close;

vis_file = [vis_dir 'pose_proj.avi'];
FrameRate = 5;
if ~exist(vis_file,'file')
    % intialize video writer
    v = VideoWriter(vis_file);
    v.FrameRate = FrameRate;
    % open new video
    open(v);
    for i = 1:N
        % read image
        file_im = [vis_dir sprintf('%05d.png',i)];
        im = imread(file_im);
        writeVideo(v,im);
    end
    % close video
    close(v);
end