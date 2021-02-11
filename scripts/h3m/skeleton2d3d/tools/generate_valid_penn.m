
frame_root = './external/Penn_Action/frames/';
label_root = './external/Penn_Action/labels/';

valid_ratio = 0.092;
output_file = './data/penn-crop/valid_ind.txt';

list_seq = dir([label_root '*.mat']);
list_seq = {list_seq.name}';
num_seq = numel(list_seq);

% reading annotations
action = cell(num_seq,1);
train = zeros(num_seq,1);
num_fr = zeros(num_seq,1);

fprintf('reading annotations ... \n');
for i = 1:num_seq
    tic_print(sprintf('%04d/%04d\n',i,num_seq));
    
    [~,name_seq] = fileparts(list_seq{i});
    fr_dir = [frame_root name_seq '/'];
    list_fr = dir([fr_dir '*.jpg']);
    list_fr = {list_fr.name}';
    num_fr(i) = numel(list_fr);
    
    lb_file = [label_root list_seq{i}];
    anno = load(lb_file);
    
    assert(ischar(anno.action));
    action{i} = anno.action;
    
    assert(anno.train == 1 || anno.train == -1);
    train(i) = anno.train;
    
    % do not count frames with no visible joints
    % remove these frames from train/valid/test data
    % 1. all joints
    % num_fr(i) = num_fr(i) - sum(all(anno.visibility == 0,2));
    % 2. difficult joints, i.e. idx 4 to 13
    num_fr(i) = num_fr(i) - sum(all(anno.visibility(:,4:end) == 0,2));
end
fprintf('\n');

% remove seq with no visible joints; set train to -2
% this is reduncdant as num_fr has been set to 0 already
rm_id = find(num_fr == 0);
train(rm_id) = -2;

% show action list
[list_act,~,ia] = unique(action, 'stable');
num_act = numel(list_act);
fprintf('action list:\n');
for i = 1:numel(list_act)
    fprintf('  %02d %s\n',i,list_act{i});
end
fprintf('\n');

% show number statistics for actions
num_seq_act_tr = zeros(num_act,1);
num_seq_act_ts = zeros(num_act,1);
num_fr_act_tr = zeros(num_act,1);
num_fr_act_ts = zeros(num_act,1);
for i = 1:numel(list_act)
    num_seq_act_tr(i) = sum(ia == i & train == 1);
    num_seq_act_ts(i) = sum(ia == i & train == -1);
    num_fr_act_tr(i) = sum(num_fr(ia == i & train == 1));
    num_fr_act_ts(i) = sum(num_fr(ia == i & train == -1));
end

% get valid ind
%   1. get number of validation seq
%   2. sample the last N training seq
num_seq_act_vl = round(num_seq_act_tr * valid_ratio);
num_seq_act_tr = num_seq_act_tr - num_seq_act_vl;
valid_ind = [];
num_fr_act_vl = zeros(num_act,1);
for i = 1:numel(list_act)
    % sample valid ind
    tr_ind = find(ia == i & train == 1);
    vl_ind = tr_ind(end-num_seq_act_vl(i)+1:end);
    valid_ind = [valid_ind; vl_ind];
    % update num_ft_act
    num_fr_act_tr(i) = num_fr_act_tr(i) - sum(num_fr(vl_ind));
    num_fr_act_vl(i) = sum(num_fr(vl_ind));
end

% show number of sequences
fprintf('number of seq/fr:\n');
fprintf('  training:    %4d/%6d\n',sum(num_seq_act_tr),sum(num_fr_act_tr));
fprintf('  validation:  %4d/%6d\n',sum(num_seq_act_vl),sum(num_fr_act_vl));
fprintf('  test:        %4d/%6d\n',sum(num_seq_act_ts),sum(num_fr_act_ts));
fprintf('  total:       %4d/%6d\n',num_seq,sum(num_fr));
fprintf('\n');

% write to file
valid_ind = num2cell(valid_ind);
valid_ind = cellfun(@(x)num2str(x),valid_ind,'UniformOutput',false);
if ~exist(output_file,'file')
    write_file_lines(output_file,valid_ind);
end
