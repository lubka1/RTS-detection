sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_iou_score',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5
    },
    'parameters': {
        'lr': {
            'values': [0.0001, 0.0002, 0.0005]
        },
        'batch_size': {
            'values': [16, 32]
        },
        'alpha': {
            'values': [0.7, 0.9]
        },
        'gamma': {
            'values': [2.0, 3.0]
        }
    }
}