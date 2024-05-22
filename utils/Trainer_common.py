
# fix the batchnormalizatio layer
def fix_bn(module):
    for m in module.modules():
        classname= m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()    # set the batchnorm layer to eval mode
    return module