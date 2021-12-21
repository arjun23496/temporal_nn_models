from options import Options
from video_prediction_model import Model

def train():
    opt = Options().parse()
    model = Model(opt)

    model.train()


if __name__ == '__main__':
    train()