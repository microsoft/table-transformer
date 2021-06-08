class Args:
    lr=1e-5
    lr_backbone=1e-5
    batch_size=5
    weight_decay=1e-4
    epochs=20
    lr_drop=1
    lr_gamma=0.9
    clip_max_norm=0.1
    
    # Model parameters
    backbone='resnet18'
    num_classes=6 # 2 for detection
    dilation=False # "If true, we replace stride with dilation in the last convolutional block (DC5)"
    position_embedding="sine"
    emphasized_weights = {}
    
    # * Transformer
    enc_layers = 6 # "Number of encoding layers in the transformer"
    dec_layers = 6 # "Number of decoding layers in the transformer"
    dim_feedforward=2048
    hidden_dim=256 # "Size of the embeddings (dimension of the transformer)"
    dropout=0.1 # "Dropout applied in the transformer"
    nheads=8 # "Number of attention heads inside the transformer's attentions"
    num_queries=125 # "Number of query slots" - Used 15 for detection
    pre_norm=True
    
    # * Segmentation
    masks=False #"Train segmentation head if the flag is provided")

    # Loss
    aux_loss=False # "Disables auxiliary decoding losses (loss at each layer)"
    
    # * Loss coefficients
    mask_loss_coef= 1 # default=1
    dice_loss_coef = 1 # default=1
    ce_loss_coef = 1 # default=1
    bbox_loss_coef=5 # default=5
    giou_loss_coef=2 # default=2
    eos_coef=0.4 # 0.1 "Relative classification weight of the no-object class")
    
    # * Matcher
    set_cost_class=1 # "Class coefficient in the matching cost"
    set_cost_bbox=5 # "L1 box coefficient in the matching cost"
    set_cost_giou=2 # "giou box coefficient in the matching cost"
    
    device='cuda'
    seed=42
    start_epoch=0
    num_workers=1
