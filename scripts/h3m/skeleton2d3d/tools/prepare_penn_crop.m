
frame_root = './external/Penn_Action/frames/';
label_root = './external/Penn_Action/labels/';

frdata_root = './data/penn-crop/frames/';
lbdata_root = './data/penn-crop/labels/';

% get list of sequences
list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

% make directory
makedir(lbdata_root);

% reading annotations
fprintf('preparing cropped dataset ... \n');
for i = 1:num_seq
    tic_print(sprintf('  %04d/%04d\n',i,num_seq));
    % read frames in sequence
    [~,name_seq] = fileparts(list_seq{i});
    fr_dir = [frame_root name_seq '/'];
    list_fr = dir([fr_dir '*.jpg']);
    list_fr = {list_fr.name}';
    % load annotation
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    assert(anno.nframes == numel(list_fr));
    % set cropped frame dir
    frdata_dir = [frdata_root name_seq '/'];
    makedir(frdata_dir);
    % get crop window
    x1 = round(min(anno.bbox(:,1)));
    y1 = round(min(anno.bbox(:,2)));
    x2 = round(max(anno.bbox(:,3)));
    y2 = round(max(anno.bbox(:,4)));
    % generate cropped frame
    for j = 1:anno.nframes
        % skip if cropped frame exists
        frdata_file = [frdata_dir list_fr{j}];
        if exist(frdata_file,'file')
            continue
        end
        % read image
        file_im = [fr_dir list_fr{j}];
        im = imread(file_im);
        im = im(y1:y2,x1:x2,:);
        imwrite(im,frdata_file);
    end
    % generate cropped label
    % skip if cropped label exists
    lbdata_file = [lbdata_root list_seq{i}];
    if exist(lbdata_file,'file')
        continue
    end
    anno.x = anno.x - x1 + 1;
    anno.y = anno.y - y1 + 1;
    anno.bbox(:,[1 3]) = anno.bbox(:,[1 3]) - x1 + 1;
    anno.bbox(:,[2 4]) = anno.bbox(:,[2 4]) - y1 + 1;
    anno.dimensions(1) = y2 - y1 + 1;
    anno.dimensions(2) = x2 - x1 + 1;
    save(lbdata_file,'-struct','anno');
end
fprintf('done.\n');
