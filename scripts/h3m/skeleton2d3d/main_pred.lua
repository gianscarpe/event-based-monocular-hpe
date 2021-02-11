-- main_pred.lua is intended to run on penn-crop only

require 'cutorch'

local DataLoader = require 'lib/dataloader'
local models = require 'lib/models/init'
local Trainer = require 'lib/train'
local opts = require 'lib/opts'

local opt = opts.parse(arg)

cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load model
local model, criterion = models.setup(opt, nil)

-- Data loading
local loaders = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, nil)

-- Compute error and accuracy
if opt.hg then
  trainer:test(0, 0, loaders, 'train')
  trainer:test(0, 0, loaders, 'val')
end

-- Predict with the final model
trainer:predict(loaders, 'train')
trainer:predict(loaders, 'val')