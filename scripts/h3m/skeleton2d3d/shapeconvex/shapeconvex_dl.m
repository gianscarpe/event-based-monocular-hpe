
% no need to reset random seed to ensure reproducibility

addpath('shapeconvex/release/ssr');
addpath('shapeconvex/release/utils');

split = 'train';

K = 128;
lam = 20;

skel.tree = struct('name',[],'children',[],'color',[]);
skel.tree(1).name = 'Hips';          skel.tree(1).children = [2 5 8];    skel.tree(1).color = 'b';
skel.tree(2).name = 'RightUpLeg';    skel.tree(2).children = 3;          skel.tree(2).color = 'r';
skel.tree(3).name = 'RightLeg';      skel.tree(3).children = 4;          skel.tree(3).color = 'r';
skel.tree(4).name = 'RightFoot';     skel.tree(4).children = [];         skel.tree(4).color = 'r';
skel.tree(5).name = 'LeftUpLeg';     skel.tree(5).children = 6;          skel.tree(5).color = 'g';
skel.tree(6).name = 'LeftLeg';       skel.tree(6).children = 7;          skel.tree(6).color = 'g';
skel.tree(7).name = 'LeftFoot';      skel.tree(7).children = [];         skel.tree(7).color = 'g';
skel.tree(8).name = 'Spine1';        skel.tree(8).children = 9;          skel.tree(8).color = 'b';
skel.tree(9).name = 'Neck';          skel.tree(9).children = [10 12 15]; skel.tree(9).color = 'b';
skel.tree(10).name = 'Head';         skel.tree(10).children = 11;        skel.tree(10).color = 'b';
skel.tree(11).name = 'Site';         skel.tree(11).children = [];        skel.tree(11).color = 'b';
skel.tree(12).name = 'LeftArm';      skel.tree(12).children = 13;        skel.tree(12).color = 'g';
skel.tree(13).name = 'LeftForeArm';  skel.tree(13).children = 14;        skel.tree(13).color = 'g';
skel.tree(14).name = 'LeftHand';     skel.tree(14).children = [];        skel.tree(14).color = 'g';
skel.tree(15).name = 'RightArm';     skel.tree(15).children = 16;        skel.tree(15).color = 'r';
skel.tree(16).name = 'RightForeArm'; skel.tree(16).children = 17;        skel.tree(16).color = 'r';
skel.tree(17).name = 'RightHand';    skel.tree(17).children = [];        skel.tree(17).color = 'r';
skel.torso = [1 2 5 9];

dataset = load(['./data/h36m/' split '.mat']);

S_train = reshape(dataset.coord_w,[size(dataset.coord_w,1),3,17]);
S_train = permute(S_train,[2 1 3]);
S_train = reshape(S_train,[3*size(S_train,2) 17]);
S_train = double(S_train);

[B, mu, ERR, SP] = learnPoseDict(S_train, skel, K, lam);

save shapeconvex/shapeDict_h36m B mu skel