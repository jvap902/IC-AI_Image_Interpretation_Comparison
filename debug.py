import timm

model = timm.create_model('aimv2_1b_patch14_336.apple_pt', pretrained=False)

print(model.default_cfg)