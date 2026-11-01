import resnet12 as models


# Only the WideResNet-50 backbone is used in FDCI-AD. The run scripts
# (shell/run-mvtec.sh, shell/run-visa.sh) always call `-b wideresnet50`.
# All other backbones from the original GLASS codebase were removed as dead code.
_BACKBONES = {
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
}


def load(name):
    return eval(_BACKBONES[name])
