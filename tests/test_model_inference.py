import torch


def test_guide_decoder(guide_decoder_1, guide_decoder_2, train_batch):
    l, g, c = train_batch
    b = l.shape[0]
    x_1 = torch.zeros(b, 256, 16, 16)
    x_2 = torch.zeros(b, 512, 16, 16)

    x = guide_decoder_1(x_1)
    assert x.shape == g.shape

    x = guide_decoder_2(x_2)
    assert x.shape == c.shape


def test_vgg(vgg, train_batch):
    _, _, c = train_batch
    x = vgg(c)
    assert list(x.shape) == [2, 4096]


def test_gen(unet_gen, vgg, guide_decoder_1, guide_decoder_2, train_batch):
    l, g, c = train_batch
    f = vgg(c)
    x, e4, d4 = unet_gen(l, f)
    assert x.shape == c.shape
    x = guide_decoder_1(e4)
    assert x.shape == g.shape
    x = guide_decoder_2(d4)
    assert x.shape == c.shape


def test_disc(disc, train_batch):
    _, _, c = train_batch
    x = disc(c)
    assert list(x.shape) == [c.shape[0], 1]
