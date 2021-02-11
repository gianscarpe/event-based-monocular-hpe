function [ pt_new ] = transform( obj, pt, center, scale, rot, res, invert, rnd )

if nargin < 7
    invert = false;
end

pt_ = ones(3,1);
pt_(1) = pt(1);
pt_(2) = pt(2);

T = getTransform(obj, center, scale, rot, res);
if invert
    T = inv(T);
end

pt_new = T * pt_;
pt_new = pt_new(1:2,:) + 1e-4;
if ~exist('rnd','var') || rnd
    pt_new = round(pt_new);
end

end