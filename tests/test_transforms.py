def test_transforms(transforms, image_batch, train_batch):
    l, g, c = train_batch
    line, gray, color = image_batch
    line, gray, color = transforms(line, gray, color)
    assert list(line.shape) == list(l[0].shape)
    assert list(gray.shape) == list(g[0].shape)
    assert list(color.shape) == list(c[0].shape)
