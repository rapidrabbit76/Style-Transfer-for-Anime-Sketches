import os
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.optim as optim
from tqdm import tqdm

from models import VGG19, Discriminator, GuideDecoder, UnetGenerator
from dataset import build_dataloader, Transforms
from utils import Mode, build_log_image, logits_2_prob
import wandb

# Line , Gray, Color
batch_type = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def training(args):
    global device, hp, gs, logger
    logger = wandb.init(config=args, save_code=True)
    gs = 0
    hp = args
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ################ dataset ################
    train_transforms = Transforms(args.image_size, train=True)
    test_transforms = Transforms(args.image_size, train=False)
    train_dataloader = build_dataloader(args, train_transforms, mode=Mode.TRAIN)
    test_dataloader = build_dataloader(args, test_transforms, mode=Mode.TEST)
    test_batch = next(iter(test_dataloader))
    line, gray, color = test_batch
    line = line.to(device)
    gray = gray.to(device)
    color = color.to(device)
    test_batch = (line, gray, color)

    ################  model  ################
    V = VGG19().to(device)
    D = Discriminator(3).to(device)
    G_f = UnetGenerator(1, 3).to(device)
    G_g1 = GuideDecoder(256, 1).to(device)
    G_g2 = GuideDecoder(512, 3).to(device)

    ################  optim  ################
    gen_optim = optim.Adam(
        params=[
            {"params": G_f.parameters()},
            {"params": G_g1.parameters()},
            {"params": G_g2.parameters()},
        ],
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
    )
    disc_optim = optim.Adam(
        params=D.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
    )
    ###############  logger  ################
    logger.watch(D, log_freq=10)
    logger.watch(G_f, log_freq=10)
    logger.watch(G_g1, log_freq=10)
    logger.watch(G_g2, log_freq=10)

    ############### training ################
    for epoch in range(args.epochs):
        training_loop(
            V,
            D,
            G_f,
            G_g1,
            G_g2,
            gen_optim,
            disc_optim,
            epoch,
            train_dataloader,
            test_batch,
        )

    ############ artifacts loging ###########
    G_f = G_f.eval()
    V = V.eval()
    G_f = torch.jit.script(G_f)
    V = torch.jit.script(V)
    gen_path = os.path.join(logger.dir, "G_f.pt.zip")
    v_path = os.path.join(logger.dir, "V.pt.zip")
    torch.jit.save(G_f, gen_path)
    torch.jit.save(V, v_path)

    if args.upload_artifacts:
        artifacts = wandb.Artifact("Style-Anime", type="model")
        artifacts.add_file(gen_path, "gen")
        artifacts.add_file(v_path, "vgg")
        logger.log_artifact(artifacts)


def training_loop(
    V: nn.Module,
    D: nn.Module,
    G_f: nn.Module,
    G_g1: nn.Module,
    G_g2: nn.Module,
    gen_optim: optim.Optimizer,
    disc_optim: optim.Optimizer,
    epoch: int,
    train_dataloader,
    test_batch: batch_type,
):
    global gs
    pbar = tqdm(train_dataloader)
    for batch_idx, batch in enumerate(pbar):
        training_info = training_step(
            batch, V, D, G_f, G_g1, G_g2, gen_optim, disc_optim
        )
        if batch_idx % 10 == 0:
            log_dict = {
                "gan/real_prob": logits_2_prob(training_info["real"]),
                "gan/fake_prob": logits_2_prob(training_info["fake"]),
                "disc/real_loss": training_info["real_loss"],
                "disc/fake_loss": training_info["fake_loss"],
                "disc/ac_loss": training_info["ac_loss"],
                "disc/loss": training_info["disc_loss"],
                "gen/output_prob": logits_2_prob(training_info["output"]),
                "gen/output_loss": training_info["output_loss"],
                "gen/recon_loss": training_info["recon_loss"],
                "gen/gd_1_loss": training_info["gd_1_loss"],
                "gen/gd_2_loss": training_info["gd_2_loss"],
                "gen/gen_loss": training_info["gen_loss"],
            }
            logger.log(log_dict, step=gs)
            pbar.set_description_str(
                (
                    f"[E:{str(epoch).zfill(3)}] "
                    f"[GS:{str(gs).zfill(8)} "
                    f"[RP:{log_dict['gan/real_prob']: 0.2f}]"
                    f"[FP:{log_dict['gan/fake_prob']: 0.2f}]"
                    f"[AC:{log_dict['disc/ac_loss']: 0.2f}]"
                    f"[GD1:{log_dict['gen/gd_1_loss']: 0.2f}]"
                    f"[GD2:{log_dict['gen/gd_2_loss']: 0.2f}]"
                    f"[GR:{log_dict['gen/recon_loss']: 0.2f}]"
                )
            )

        if batch_idx % 100 == 0:
            test_info = test_step(test_batch, V, G_f, G_g1, G_g2)
            log_dict = {
                "test/recon_loss": test_info["recon_loss"],
                "test/gd_1_loss": test_info["gd_1_loss"],
                "test/gd_2_loss": test_info["gd_2_loss"],
            }
            logger.log(log_dict, step=gs)

            train_image = build_log_image(
                [
                    training_info["line"],
                    training_info["gray"],
                    training_info["gd_gray"],
                    training_info["gd_color"],
                    training_info["_color"],
                    training_info["color"],
                ],
            )
            test_image = build_log_image(
                [
                    test_info["line"],
                    test_info["gray"],
                    test_info["gd_gray"],
                    test_info["gd_color"],
                    test_info["_color"],
                    test_info["color"],
                ],
            )
            logger.log(
                {
                    "image/train-sample": wandb.Image(train_image),
                    "image/test-sample": wandb.Image(test_image),
                },
                step=gs,
            )


def training_step(
    batch: batch_type,
    V: nn.Module,
    D: nn.Module,
    G_f: nn.Module,
    G_g1: nn.Module,
    G_g2: nn.Module,
    gen_optim: optim.Optimizer,
    disc_optim: optim.Optimizer,
) -> Dict[str, torch.Tensor]:
    torch.set_grad_enabled(True)
    global device, gs
    line, gray, color = [d.to(device) for d in batch]

    #### Discriminator feed forward ####
    global_hint = V(color)
    _color, e4, d4 = G_f(line, global_hint)
    disc_real = D(color)
    disc_fake = D(_color.detach())
    real_loss = F.binary_cross_entropy_with_logits(
        disc_real, torch.ones_like(disc_real)
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        disc_fake, torch.zeros_like(disc_fake)
    )
    ac_loss = torch.mean(torch.ones_like(global_hint) - global_hint)
    disc_loss = real_loss + ac_loss + fake_loss

    #### Discriminator update ####
    disc_optim.zero_grad()
    disc_loss.backward()
    disc_optim.step()

    #### Generator feed forward ####
    gd_gray = G_g1(e4)
    gd_color = G_g2(d4)
    output = D(_color)
    output_loss = F.binary_cross_entropy_with_logits(
        output, torch.ones_like(output)
    )
    recon_loss = F.l1_loss(_color, color)
    gd_1_loss = F.l1_loss(gd_gray, gray)
    gd_2_loss = F.l1_loss(gd_color, color)
    gen_loss = (
        output_loss
        + (recon_loss + gd_1_loss * hp.alpha + gd_2_loss * hp.beta) * hp.gamma
    )

    #### Generator update ####
    gen_optim.zero_grad()
    gen_loss.backward()
    gen_optim.step()

    gs += 1
    torch.set_grad_enabled(False)
    return {
        # image
        "line": line,
        "gray": gray,
        "color": color,
        "gd_gray": gd_gray,
        "gd_color": gd_color,
        "_color": _color,
        # scalar
        "real": disc_real,
        "fake": disc_fake,
        "real_loss": real_loss,
        "fake_loss": fake_loss,
        "ac_loss": ac_loss,
        "disc_loss": disc_loss,
        "output": output,
        "output_loss": output_loss,
        "recon_loss": recon_loss,
        "gd_1_loss": gd_1_loss,
        "gd_2_loss": gd_2_loss,
        "gen_loss": gen_loss,
    }


def test_step(
    batch: batch_type,
    V: nn.Module,
    G_f: nn.Module,
    G_g1: nn.Module,
    G_g2: nn.Module,
) -> Dict[str, torch.Tensor]:
    line, gray, color = batch
    global_hint = V(color)
    _color, e4, d4 = G_f(line, global_hint)
    gd_gray = G_g1(e4)
    gd_color = G_g2(d4)
    recon_loss = F.l1_loss(_color, color)
    gd_1_loss = F.l1_loss(gd_gray, gray)
    gd_2_loss = F.l1_loss(gd_color, color)
    return {
        # image
        "line": line,
        "gray": gray,
        "color": color,
        "gd_gray": gd_gray,
        "gd_color": gd_color,
        "_color": _color,
        # scalar
        "recon_loss": recon_loss,
        "gd_1_loss": gd_1_loss,
        "gd_2_loss": gd_2_loss,
    }
