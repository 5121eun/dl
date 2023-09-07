import torch

def get_giou(gt_b, pd_b):
    
    tg_x1, tg_y1, tg_x2, tg_y2 = gt_b.unbind(dim=-1)
    sc_x1, sc_y1, sc_x2, sc_y2 = pd_b.unbind(dim=-1)

    x1 = torch.stack([tg_x1, sc_x1], dim=-1)
    x2 = torch.stack([tg_x2, sc_x2], dim=-1)

    y1 = torch.stack([tg_y1, sc_y1], dim=-1)
    y2 = torch.stack([tg_y2, sc_y2], dim=-1)
    
    inter_x1 = torch.max(x1, dim=-1).values
    inter_x2= torch.min(x2, dim=-1).values

    inter_y1 = torch.max(y1, dim=-1).values
    inter_y2= torch.min(y2, dim=-1).values
    
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0
    
    inter = inter_w * inter_h
    
    gt_b_area = (tg_x2 - tg_x1) * (tg_y2 - tg_y1)
    pd_b_area = (sc_x2 - sc_x1) * (sc_y2 - sc_y1)

    union = (gt_b_area + pd_b_area) - inter
    iou = inter / union

    b_x1 = torch.min(x1, dim=-1).values
    b_x2= torch.max(x2, dim=-1).values

    b_y1 = torch.min(y1, dim=-1).values
    b_y2= torch.max(y2, dim=-1).values
    
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    return 1 - (iou - ((b_area - union) / b_area))

def xyxy_to_cxcywh(b):
    x1, y1, x2, y2  = b.unbind(-1)
    
    w = x2 - x1
    h = y2 - y1
    
    cx = x1 + w/2
    cy = y1 + h/2

    return torch.stack((cx, cy, w, h), -1)

def xywh_to_cxcywh(b):
    x, y, w, h  = b.unbind(-1)
    
    cx = x + w/2
    cy = y + h/2
    
    return torch.stack((cx, cy, w, h), -1)

def cxcywh_to_xywh(b):
    cx, cy, w, h  = b.unbind(-1)
    
    x = cx - w/2
    y = cy - h/2
    
    return torch.stack((x, y, w, h), -1)