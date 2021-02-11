function [ totalImg ] = compileImages( obj, imgs, nrows, ncols, res )

% Assumes the input images are all square/the same resolution
totalImg = zeros(3,nrows*res,ncols*res);
for i = 1:numel(imgs)
    r = floor((i-1)/ncols)+1;
    c = mod((i-1),ncols)+1;
    totalImg(:,(r-1)*res+1:r*res,(c-1)*res+1:c*res) = imgs{i};
end

end