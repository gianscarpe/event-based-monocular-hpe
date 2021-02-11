classdef img
    methods
        T = getTransform(obj, center, scale, rot, res);
        
        pt_new = transform(obj, pt, center, scale, rot, res, invert, rnd);
        
        img = crop(obj, img, center, scale, rot, res);
        
        img = drawGaussian(obj, img, pt, sigma);
        
        cl = colorHM(obj, x);
        
        totalImg = compileImages(obj, imgs, nrows, ncols, res);
    end
end