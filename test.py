from options import Options
from model import Model

def test():
    opt = Options().parse()
    assert opt.pretrained_model != ''
    model = Model(opt)

    model.test()


if __name__ == '__main__':
    test()