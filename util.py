import gym
import torch
import numpy as np
import torchvision.transforms as T
from itertools import count
import matplotlib
import matplotlib.pyplot as plt


# Make the input image grayscale
convert = T.Compose([T.ToPILImage(),
                    T.Grayscale(),
                    T.ToTensor()])


def to_tensor(state, device):
    # Convert to float, convert to torch tensor
    state = np.ascontiguousarray(state, dtype=np.float32) / 255
    state = torch.from_numpy(state)
    # Convert the input image to grayscale
    state = convert(state)
    # Add a batch dimension (BCHW)
    return state.unsqueeze(0).to(device)


def show_img(screen, grayscale):
    plt.figure()
    if grayscale:
        screen = screen.cpu().squeeze(0).numpy()
        plt.imshow(screen[0])
    else:
        plt.imshow(screen.transpose((1, 2, 0)), interpolation='none')
    plt.show()


def generateDataset(env, eposides, list, device):
    for i_episode in range(eposides):
        env.reset()
        state = env.render(mode='rgb_array').transpose((2, 0, 1))
        state = to_tensor(state, device)

        for t in count():
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            action = torch.tensor([action], device=device)
            reward = torch.tensor([reward], device=device)

            next_state = next_state.transpose((2, 0, 1))
            next_state = to_tensor(next_state, device)

            list.append((state, action, next_state, reward))

            state = next_state


# env = gym.make('Asteroids-v0')
# EPOSIDE = 1
# dataset = []
#
# # if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# generateDataset(env, EPOSIDE, dataset, device)
#
# for i in range(len(dataset)):
#     state = dataset[i][0]
#     next_state = dataset[i][2]
#
#     show_img(state, True)
#     show_img(next_state, True)







