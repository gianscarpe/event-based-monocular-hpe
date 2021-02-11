function [ img ] = crop( obj, img, center, scale, rot, res )

ul = transform(obj, [1,1], center, scale, 0, res, true);
br = transform(obj, [res+1,res+1], center, scale, 0, res, true);

% pad = floor(norm(ul-br)/2 - br(1)-ur(1)/2);

if rot ~= 0
    error('not implemented yet.')
end

if numel(size(img)) > 2
    newDim = [size(img,1); br(2)-ul(2); br(1)-ul(1)];
    newImg = zeros(newDim(1), newDim(2), newDim(3));
    ht = size(img,2);
    wd = size(img,3);
else
    newDim = [br(2)-ul(2); br(1)-ul(1)];
    newImg = zeros(newDim(1),newDim(2));
    ht = size(img,1);
    wd = size(img,2);
end

newX = [max(1, -ul(1)+2), min(br(1),wd+1)-ul(1)];
newY = [max(1, -ul(2)+2), min(br(2),ht+1)-ul(2)];
oldX = [max(1, ul(1)), min(br(1),wd+1)-1];
oldY = [max(1, ul(2)), min(br(2),ht+1)-1];

if size(newDim,1) > 2
    newImg(1:newDim(1),newY(1):newY(2),newX(1):newX(2)) = img(1:newDim(1),oldY(1):oldY(2),oldX(1):oldX(2));
else
    newImg(newY(1):newY(2),newX(1):newX(2)) = img(oldY(1):oldY(2),oldX(1):oldX(2));
end

if rot ~= 0
    error('not implemented yet.')
end

newImg = permute(newImg, [2 3 1]);
newImg = imresize(newImg,[res res]);
img = permute(newImg, [3 1 2]);

end