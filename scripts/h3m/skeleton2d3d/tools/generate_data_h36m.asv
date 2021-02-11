
H36MDataBase.instance();

Features{1} = H36MPose3DPositionsFeature();

% set parameters
part = 'body';
samp = 10;

im_root = 'data/h36m/frames/';
data_pose = 'data/h36m/frames/';
% training set
S = [1 7 8 9 11];

skip_id = [11 2 2];

data_file = './data/h36m/train.mat';


for s = [1]
    for a = 2:16
        for b = 1:2
            for c = 1:4
            tt = tic;
            fprintf('  subject %02d, action %02d-%d',s,a,b);
            path = sprintf('data/full/S%01d/MyPoseFeatures/D3_positions/', s);
            fprintf(Sequence.Name);
            
            makedir(path)
            Sequence = H36MSequence(s, a, b, c);
            F = H36MComputeFeatures(Sequence, Features);
            if iscell(F)
            save(sprintf('%s%s.mat', path, Sequence.BaseName), 'F');
            end
            end
        end
    end
end



fprintf('generate training data ... \n');
ind2sub = int32(zeros(0,4));
coord_w = single(zeros(0,51));
coord_c = single(zeros(4,0,51));
coord_p = single(zeros(4,0,34));
focal = single(zeros(4,2));
for s = S
    for a = 2:16
        for b = 1:2
            tt = tic;
            fprintf('  subject %02d, action %02d-%d',s,a,b);

            c = 1;
            Sequence = H36MSequence(s, a, b, c);
            F = H36MComputeFeatures(Sequence, Features);
            Subject = Sequence.getSubject();
            posSkel = Subject.getPosSkel();
            [pose, posSkel] = Features{1}.select(F{1}, posSkel, part);
            
            sid = round((samp-1)/2)+1;
            ind = sid:samp:size(pose,1);
            pose = pose(ind,:);

            ind2sub = [ind2sub; [repmat([s a b],[numel(ind) 1]) ind']];  %#ok
            coord_w = [coord_w; pose];  %#ok

            for c = 1:4
                Sequence = H36MSequence(s, a, b, c);
                Camera = Sequence.getCamera();
                
                % assume same camera has same parameters for different sequences 
                if focal(c,1) == 0 && focal(c,2) == 0
                    focal(c,:) = Camera.f;
                else
                    assert(abs(focal(c,1) - Camera.f(1)) < 1e-4);
                    assert(abs(focal(c,2) - Camera.f(2)) < 1e-4);
                end
                
                % get projection and crop box
                cc = single(zeros(0,51));
                cp = single(zeros(0,34));
                bbox = zeros(size(pose,1),4);
                for p = 1:size(pose,1)
                    P = reshape(pose(p,:),[3 numel(pose(p,:))/3])';
                    N = size(P,1);
                    X = Camera.R*(P'-Camera.T'*ones(1,N));
                    cc = [cc; reshape(X,[1 numel(X)])];  %#ok

                    proj = Camera.project(Features{1}.toPositions(pose(p,:),posSkel));
                    
                    minx = min(proj(1:2:33));
                    maxx = max(proj(1:2:33));
                    miny = min(proj(2:2:34));
                    maxy = max(proj(2:2:34));
                    rx = max(abs([minx,maxx] - Camera.c(1)));
                    ry = max(abs([miny,maxy] - Camera.c(2)));
                    r = max(rx,ry);
                    bbox(p,1) = round(Camera.c(1) - r);
                    bbox(p,2) = round(Camera.c(1) + r);
                    bbox(p,3) = round(Camera.c(2) - r);
                    bbox(p,4) = round(Camera.c(2) + r);
                    proj(1:2:33) = proj(1:2:33) - bbox(p,1) + 1;
                    proj(2:2:34) = proj(2:2:34) - bbox(p,3) + 1;
                    cp = [cp; proj'];  %#ok
                end
                
                if c == 1
                    sz = size(coord_c,2);
                end
                coord_c(c,sz+1:sz+size(cc,1),:) = cc;
                coord_p(c,sz+1:sz+size(cc,1),:) = cp;
                
                % skip corrupted videos 
                if ismember([s a b],skip_id,'rows')
                    continue
                end
                
                vidfeat = H36MRGBVideoFeature();
                da = vidfeat.serializer(Sequence);
                assert(da.Reader.NumberOfFrames >= size(pose,1));
                assert(da.Reader.Height == Camera.Resolution(1));
                assert(da.Reader.Width == Camera.Resolution(2));
                
                im_dir = [im_root sprintf('%02d/%02d/%1d/',s,a,b)];
                makedir(im_dir);
                
                for p = 1:size(pose,1)
                    im_file = [im_dir sprintf('%1d_%04d.jpg',c,ind(p))];
                    if ~exist(im_file,'file')
                        x1 = bbox(p,1);
                        x2 = bbox(p,2);
                        y1 = bbox(p,3);
                        y2 = bbox(p,4);
                        padx1 = max(0,1-x1);
                        padx2 = max(0,x2-Camera.Resolution(2));
                        pady1 = max(0,1-y1);
                        pady2 = max(0,y2-Camera.Resolution(1));
                        x1_c = padx1+1;
                        x2_c = x2-x1+1-padx2;
                        y1_c = pady1+1;
                        y2_c = y2-y1+1-pady2;
                        x1_o = x1+padx1;
                        x2_o = x2-padx2;
                        y1_o = y1+pady1;
                        y2_o = y2-pady2;
                        im = da.getFrame(ind(p));
                        im_crop = uint8(zeros(y2-y1+1,x2-x1+1,3));
                        im_crop(y1_c:y2_c,x1_c:x2_c,:) = im(y1_o:y2_o,x1_o:x2_o,:);
                        imwrite(im_crop,im_file);
                    end
                end
            end
            
            time = toc(tt);
            fprintf('  %7.2f sec.\n',time);
        end
    end
end
% need version -V6 for ilcomp
if ~exist(data_file,'file')
    save(data_file,'ind2sub','coord_w','coord_c','coord_p','focal','-V6');
end
fprintf('done.\n');

% validation set
S = [5 6];

skip_id = zeros(0,3);

data_file = './data/h36m/val.mat';

fprintf('generate validation data ... \n');
ind2sub = int32(zeros(0,4));
coord_w = single(zeros(0,51));
coord_c = single(zeros(4,0,51));
coord_p = single(zeros(4,0,34));
focal = single(zeros(4,2));
for s = S
    for a = 2:16
        for b = 1:2
            tt = tic;
            fprintf('  subject %02d, action %02d-%d',s,a,b);
            
            c = 1;
            Sequence = H36MSequence(s, a, b, c);
            F = H36MComputeFeatures(Sequence, Features);
            Subject = Sequence.getSubject();
            posSkel = Subject.getPosSkel();
            [pose, posSkel] = Features{1}.select(F{1}, posSkel, part);
            
            sid = round((samp-1)/2)+1;
            ind = sid:samp:size(pose,1);
            pose = pose(ind,:);
            
            ind2sub = [ind2sub; [repmat([s a b],[numel(ind) 1]) ind']];  %#ok
            coord_w = [coord_w; pose];  %#ok
            
            for c = 1:4
                Sequence = H36MSequence(s, a, b, c);
                Camera = Sequence.getCamera();
                
                % assume same camera has same parameters for different sequences 
                if focal(c,1) == 0 && focal(c,2) == 0
                    focal(c,:) = Camera.f;
                else
                    assert(abs(focal(c,1) - Camera.f(1)) < 1e-4);
                    assert(abs(focal(c,2) - Camera.f(2)) < 1e-4);
                end
                
                % get projection and crop box
                cc = single(zeros(0,51));
                cp = single(zeros(0,34));
                bbox = zeros(size(pose,1),4);
                for p = 1:size(pose,1)
                    P = reshape(pose(p,:),[3 numel(pose(p,:))/3])';
                    N = size(P,1);
                    X = Camera.R*(P'-Camera.T'*ones(1,N));
                    cc = [cc; reshape(X,[1 numel(X)])];  %#ok

                    proj = Camera.project(Features{1}.toPositions(pose(p,:),posSkel));
                    
                    minx = min(proj(1:2:33));
                    maxx = max(proj(1:2:33));
                    miny = min(proj(2:2:34));
                    maxy = max(proj(2:2:34));
                    rx = max(abs([minx,maxx] - Camera.c(1)));
                    ry = max(abs([miny,maxy] - Camera.c(2)));
                    r = max(rx,ry);
                    bbox(p,1) = round(Camera.c(1) - r);
                    bbox(p,2) = round(Camera.c(1) + r);
                    bbox(p,3) = round(Camera.c(2) - r);
                    bbox(p,4) = round(Camera.c(2) + r);
                    proj(1:2:33) = proj(1:2:33) - bbox(p,1) + 1;
                    proj(2:2:34) = proj(2:2:34) - bbox(p,3) + 1;
                    cp = [cp; proj'];  %#ok
                end
                
                if c == 1
                    sz = size(coord_c,2);
                end
                coord_c(c,sz+1:sz+size(cc,1),:) = cc;
                coord_p(c,sz+1:sz+size(cc,1),:) = cp;
                
                % skip corrupted videos 
                if ismember([s a b],skip_id,'rows')
                    continue
                end
                
                vidfeat = H36MRGBVideoFeature();
                da = vidfeat.serializer(Sequence);
                assert(da.Reader.NumberOfFrames >= size(pose,1));
                assert(da.Reader.Height == Camera.Resolution(1));
                assert(da.Reader.Width == Camera.Resolution(2));
                
                im_dir = [im_root sprintf('%02d/%02d/%1d/',s,a,b)];
                makedir(im_dir);
                
                for p = 1:size(pose,1)
                    im_file = [im_dir sprintf('%1d_%04d.jpg',c,ind(p))];
                    if ~exist(im_file,'file')
                        x1 = bbox(p,1);
                        x2 = bbox(p,2);
                        y1 = bbox(p,3);
                        y2 = bbox(p,4);
                        padx1 = max(0,1-x1);
                        padx2 = max(0,x2-Camera.Resolution(2));
                        pady1 = max(0,1-y1);
                        pady2 = max(0,y2-Camera.Resolution(1));
                        x1_c = padx1+1;
                        x2_c = x2-x1+1-padx2;
                        y1_c = pady1+1;
                        y2_c = y2-y1+1-pady2;
                        x1_o = x1+padx1;
                        x2_o = x2-padx2;
                        y1_o = y1+pady1;
                        y2_o = y2-pady2;
                        im = da.getFrame(ind(p));
                        im_crop = uint8(zeros(y2-y1+1,x2-x1+1,3));
                        im_crop(y1_c:y2_c,x1_c:x2_c,:) = im(y1_o:y2_o,x1_o:x2_o,:);
                        imwrite(im_crop,im_file);
                    end
                end
            end
            
            time = toc(tt);
            fprintf('  %7.2f sec.\n',time);
        end
    end
end
% need version -V6 for ilcomp
if ~exist(data_file,'file')
    save(data_file,'ind2sub','coord_w','coord_c','coord_p','focal','-V6');
end
fprintf('done.\n');
