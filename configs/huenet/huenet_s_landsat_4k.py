_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py',
    '../_base_/datasets/landsat.py'
]
# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_s_20230227-f33ccdf2.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[16.4502, 19.5271, 12.6983],
    std=[7.6011, 5.6420, 5.2811],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    test_cfg=dict(size_divisor=32))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='FCA_MSCAN',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        in_channels=3,
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[2, 2, 4, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        # in_channels=[128, 320, 512],
        # in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        out_channels=2,
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000), #在线难样本挖掘 (Online Hard Example Mining, OHEM),可解决样本不平衡的问题
        # out_channels=1,
        loss_decode=
            # type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            # dict(type='FocalLoss', loss_name='loss_focal', use_sigmoid=False,loss_weight=1.0),
            # dict(type='DiceLoss', loss_name='loss_dice', use_sigmoid=False,loss_weight=3.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=20000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# dataset settings
randomness = dict(seed=0)
train_dataloader = dict(batch_size=8)
work_dir = './work_dirs/huenet_s_landsat'

