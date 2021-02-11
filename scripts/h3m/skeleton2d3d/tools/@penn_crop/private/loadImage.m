function [ im ] = loadImage( obj, idx )

im = imread(fullfile(obj.dir, imgpath(obj, idx)));
im = permute(im, [3 1 2]);

end

