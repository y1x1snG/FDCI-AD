# Third-Party Notices

FDCI-AD itself is released under the MIT License — see [LICENSE](LICENSE).

This repository redistributes code that originates from third parties. The full
license text of each such component is reproduced below, as required by the terms
of those licenses.

---

## 1. GLASS — base framework

The FDCI-AD pipeline in this repository is a derivative work of **GLASS** (Global
and Local Anomaly co-Synthesis), on top of which the FDSIA, MPDC, and EDCT modules
are added. The GLASS reference is:

> Q. Chen, H. Luo, C. Lv, and Z. Zhang, "A Unified Anomaly Synthesis Strategy with
> Gradient Ascent for Industrial Anomaly Detection and Localization,"
> arXiv:2407.09359, 2024. <https://arxiv.org/abs/2407.09359>

GLASS is distributed under the MIT License, reproduced in full below. Its
copyright notice is also retained in [LICENSE](LICENSE).

```
MIT License

Copyright (c) 2024 Qiyu Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. torchvision — vendored internal modules

The files under [`torchvision_vendored/`](torchvision_vendored/) are verbatim,
unmodified copies of internal modules of PyTorch's torchvision
(<https://github.com/pytorch/vision>), version 0.15.1:

```
torchvision_vendored/__init__.py
torchvision_vendored/___utils.py
torchvision_vendored/__utils.py
torchvision_vendored/_api.py
torchvision_vendored/_functional_pil.py
torchvision_vendored/_functional_tensor.py
torchvision_vendored/_internally_replaced_utils.py
torchvision_vendored/_meta.py
torchvision_vendored/_presets.py
torchvision_vendored/_utils.py
torchvision_vendored/functional.py
```

They are included only so that the modified WideResNet-50 backbone in `resnet12.py`
can run without patching the user's torchvision installation. They are **not** part
of FDCI-AD's own contribution. `resnet12.py` adds this directory to `sys.path` at
import time.

torchvision is distributed under the BSD 3-Clause License, reproduced in full below.

```
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 3. Datasets

MVTec AD, VisA, and DTD are **not** redistributed in this repository. They are
downloaded separately by the user under their respective licenses:

- MVTec AD — <https://www.mvtec.com/company/research/datasets/mvtec-ad>
- VisA — <https://github.com/amazon-science/spot-diff>
- DTD — <https://www.robots.ox.ac.uk/~vgg/data/dtd/>
