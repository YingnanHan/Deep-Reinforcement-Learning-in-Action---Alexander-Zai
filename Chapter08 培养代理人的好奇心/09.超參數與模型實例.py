

params = {
    'batch_size':150,
    'beta':0.2,
    'lambda':0.1,
    'eta':1.0,
    'gamma':0.2,
    'max_episode_len':15,
    'action_repeats':6,
    'frames_per_state':3
}

replay = ExperienceReplay(N=1000,batch_size=params['batch_size'])
Qmodel = Qnetwork()
encoder = Phi()
forward_model = Fnet()
inverse_model = Gnet()
forward_loss = nn.MSELoss(reduction = 'none')
inverse_loss = nn.CrossEntropyLoss(reduction = 'none')
qloss = nn.MSELoss()
all_model_params = list(Qmodel.parameters()) + list(encoder.parameters()) + list(forward_model.parameters()) + list(inverse_model.parameters())
opt = optim.Adam(lr=0.001,params=all_model_params)