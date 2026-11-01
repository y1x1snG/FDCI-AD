# Vendored internals of PyTorch's torchvision (BSD 3-Clause,
# Copyright (c) Soumith Chintala 2016). The full license text and the complete
# list of third-party notices are reproduced in THIRD_PARTY_NOTICES.md at the
# repository root.
#
# These are copies of torchvision's internal modules (transforms/functional,
# models internal helpers, etc.) that the modified WideResNet-50 backbone in
# resnet12.py depends on. They are NOT part of FDCI-AD's own contribution and are
# included only so the backbone runs without patching the user's torchvision
# install. resnet12.py adds this directory to sys.path at import time.
