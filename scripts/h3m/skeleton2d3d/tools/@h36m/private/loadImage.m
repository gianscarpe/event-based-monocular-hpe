function [ im ] = loadImage( obj, idx, cam )

im = imread(fullfile(obj.dir, imgpath(obj, idx, cam)));
im = permute(im, [3 1 2]);

end

