classdef h36m
    properties (Access = private)
        split
        dir
        anno_file
        inputRes
        outputRes
        ind2sub
        coord_w
        coord_c
        coord_p
        focal
        jointType
        numPt
        hg
        img
    end
    methods
        % constructor
        function obj = h36m(opt, split)
            obj.split = split;
            obj.dir = fullfile(opt.data, 'frames');
            assert(exist(obj.dir,'dir') == 7, ['directory does not exist: ' obj.dir]);
            obj.anno_file = fullfile(opt.data, [split '.mat']);
            obj.inputRes = opt.inputResHG;
            obj.outputRes = opt.inputRes;
            % load annotation
            anno = load(obj.anno_file);
            obj.ind2sub = anno.ind2sub;
            obj.coord_w = anno.coord_w;
            obj.coord_c = anno.coord_c;
            obj.coord_p = anno.coord_p;
            obj.focal = anno.focal;
            % convert to Penn Action's format
            if opt.penn
                obj.jointType = 'penn-crop';
                obj.coord_w = obj.coord_w(:,getPennInd(obj,3),:);
                obj.coord_c = obj.coord_c(:,:,getPennInd(obj,3),:);
                obj.coord_p = obj.coord_p(:,:,getPennInd(obj,2),:);
            else
                obj.jointType = 'h36m';
            end
            % Get number of joints
            obj.numPt = size(obj.coord_w,2) / 3;
            % check if the model contains hourglass for pose estimation
            obj.hg = opt.hg;
            % remove corrupted images
            if obj.hg
                rm = [11,2,2];
                for i = 1:size(rm,1)
                    i1 = obj.ind2sub(:,1) == rm(i,1);
                    i2 = obj.ind2sub(:,2) == rm(i,2);
                    i3 = obj.ind2sub(:,3) == rm(i,3);
                    keep = find((i1+i2+i3) ~= 3);
                    obj.ind2sub = obj.ind2sub(keep,:);
                    obj.coord_w = obj.coord_w(keep,:,:);
                    obj.coord_c = obj.coord_c(:,keep,:,:);
                    obj.coord_p = obj.coord_p(:,keep,:,:);
                end
            end
            % load lib
            obj.img = img();
        end
        
        % get image path
        ind = getPennInd(obj, dim);
        
        % load 3d pose in camera coordinates
        coord_c = loadPoseCamera(obj, idx, cam);
        
        % load focal length
        focal = loadFocal(obj, cam);
        
        % get image path
        pa = imgpath(obj, idx, cam);
        
        % load image
        im = loadImage(obj, idx, cam);
        
        % get center and scale
        [center, scale] = getCenterScale(obj, im);
        
        % get dataset size
        out = size(obj);
        
        [input, repos, trans, focal, proj] = get(obj, idx, cam);
    end
end