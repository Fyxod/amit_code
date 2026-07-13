import torch
ckpt = torch.load('checkpoints/checkpoint_0.pt', map_location='cpu', weights_only=False)
print('Keys:', list(ckpt.keys()))
print('Iteration:', ckpt['iteration'])
print('Best loss:', ckpt['best_loss'])
print('History len:', len(ckpt['history']))
print('State dict keys:', list(ckpt['model_state_dict'].keys()))
if ckpt['history']:
    print('History[0]:', ckpt['history'][0])
ckpt50 = torch.load('checkpoints/checkpoint_50.pt', map_location='cpu', weights_only=False)
print('\n--- Checkpoint 50 ---')
print('History len:', len(ckpt50['history']))
if ckpt50['history']:
    print('Last history entry:', ckpt50['history'][-1])