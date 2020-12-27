import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=64, hidden_layer=1, init_weight=True):
        super(DQN, self).__init__()
        lst = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(hidden_layer):
            lst.append(nn.Linear(hidden_dim, hidden_dim))
            lst.append(nn.ReLU())
        lst.append(nn.Linear(hidden_dim, 1))
        self.backbone = nn.Sequential(*lst)

        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)


class SimpleTetrisDQN(nn.Module):
    def __init__(self, state_size=200, info_size=14, hidden_dim=64, hidden_layer=1, act_size=40, init_weight=True):
        super(SimpleTetrisDQN, self).__init__()
        lst = [nn.Linear(state_size+info_size, hidden_dim), nn.ReLU()]
        for i in range(hidden_layer):
            lst.append(nn.Linear(hidden_dim, hidden_dim))
            lst.append(nn.ReLU())
        lst.append(nn.Linear(hidden_dim, act_size))
        self.backbone = nn.Sequential(*lst)

        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, info):
        x = torch.cat((x, info), dim=1)
        return self.backbone(x)


class SimpleTetrisConvDQN(nn.Module):
    def __init__(self, state_size=200, info_size=38, s_hid_size=64, hidden_dim=64, hidden_layer=0, act_size=40, init_weight=True):
        super(SimpleTetrisConvDQN, self).__init__()
        backbone_lst = []
        mlp_lst = []
        self.board_w = 10
        self.board_h = 20
        self.s_hid_size = s_hid_size
        self.boundary_pad = nn.ConstantPad2d(1, 1.0) # Nx1x22x12
        backbone_lst.extend([nn.Conv2d(1, 4, 3, 1), nn.ReLU()]) # Nx4x20x10
        backbone_lst.extend([nn.Conv2d(4, 8, 4, 2, padding=1), nn.ReLU()]) # Nx8x10x5
        backbone_lst.extend([nn.Conv2d(8, 16, 3, (2,1), padding=(1,1)), nn.ReLU()]) # Nx16x5x5
        backbone_lst.extend([nn.Conv2d(16, 32, 3, 1), nn.ReLU()]) # Nx32x3x3
        backbone_lst.extend([nn.Conv2d(32, s_hid_size, 3, 1), nn.ReLU()]) # Nx64x1x1
        self.backbone = nn.Sequential(*backbone_lst)

        mlp_lst.extend([nn.Linear(s_hid_size+info_size, hidden_dim), nn.ReLU()])
        for i in range(hidden_layer):
            mlp_lst.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        mlp_lst.append(nn.Linear(hidden_dim, act_size))
        self.mlp = nn.Sequential(*mlp_lst)

        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, info):
        x = x.reshape(-1, 1, self.board_h, self.board_w)
        x = self.backbone(self.boundary_pad(x)).view(-1, self.s_hid_size)
        x = torch.cat((x, info), dim=1)
        return self.mlp(x)

if __name__ == "__main__":
    from tetris import SymbolTetrisSimple
    import gym
    device = torch.device("cpu")
    env = SymbolTetrisSimple(gym.make('Tetris-v0'), 
                        max_episode_length=-1, align=False)
    n_actions = env.game_board_width * 4 # board_width * rotation
    state_size = env.game_board_width * env.game_board_height
    info_size = 2 * env.num_tetris_kind

    model = SimpleTetrisConvDQN(state_size=state_size, info_size=info_size).to(device)
    next_state, reward, done, next_info, line = env.reset()
    next_state, reward, done, next_info, line = env.step(31)
    next_state, reward, done, next_info, line = env.step(5)
    state = next_state.reshape(1, -1)
    info = next_info.reshape(1, -1)
    state = torch.FloatTensor(state).to(device)
    info = torch.FloatTensor(info).to(device)
    test = model(state, info)

    env.close()