# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import (
    AHNet,
    AttentionUnet,
    AutoEncoder,
    BasicUNet,
    BasicUNetPlusPlus,
    DenseNet121,
    DynUNet,
    EfficientNetBN,
    FlexibleUNet,
    FullyConnectedNet,
    HighResNet,
    SegResNet,
    SENet154,
    UNet,
)
from monai.utils import optional_import

# torch.compile is available in PyTorch 2.0+
torch_compile, has_torch_compile = optional_import("torch", name="compile")

# Test cases: [network_class, init_kwargs, input_shape]
TEST_CASES = [
    # UNet variants
    [
        UNet,
        {"spatial_dims": 2, "in_channels": 1, "out_channels": 2, "channels": (8, 16), "strides": (2,)},
        (2, 1, 32, 32),
    ],
    [
        UNet,
        {"spatial_dims": 3, "in_channels": 1, "out_channels": 2, "channels": (8, 16), "strides": (2,)},
        (2, 1, 16, 16, 16),
    ],
    [
        BasicUNet,
        {"spatial_dims": 2, "in_channels": 1, "out_channels": 2, "features": (8, 8, 16, 16, 32, 32)},
        (2, 1, 32, 32),
    ],
    [
        BasicUNet,
        {"spatial_dims": 3, "in_channels": 1, "out_channels": 2, "features": (8, 8, 16, 16, 32, 32)},
        (2, 1, 16, 16, 16),
    ],
    [
        BasicUNetPlusPlus,
        {"spatial_dims": 2, "in_channels": 1, "out_channels": 2, "features": (8, 16, 32, 64, 128, 8)},
        (2, 1, 32, 32),
    ],
    [
        AttentionUnet,
        {"spatial_dims": 2, "in_channels": 1, "out_channels": 2, "channels": (8, 16), "strides": (2,)},
        (2, 1, 32, 32),
    ],
    [
        FlexibleUNet,
        {
            "in_channels": 1,
            "out_channels": 2,
            "backbone": "efficientnet-b0",
            "pretrained": False,
            "spatial_dims": 2,
        },
        (2, 1, 32, 32),
    ],
    # SegResNet
    [
        SegResNet,
        {"spatial_dims": 2, "in_channels": 1, "out_channels": 2, "init_filters": 8},
        (2, 1, 32, 32),
    ],
    [
        SegResNet,
        {"spatial_dims": 3, "in_channels": 1, "out_channels": 2, "init_filters": 8},
        (2, 1, 16, 16, 16),
    ],
    # DynUNet
    [
        DynUNet,
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 2,
            "kernel_size": [[3, 3], [3, 3], [3, 3]],
            "strides": [[1, 1], [2, 2], [2, 2]],
            "upsample_kernel_size": [[2, 2], [2, 2]],
        },
        (2, 1, 32, 32),
    ],
    # HighResNet
    [
        HighResNet,
        {"spatial_dims": 2, "in_channels": 1, "out_channels": 2},
        (2, 1, 32, 32, 32),
    ],
    # AutoEncoder
    [
        AutoEncoder,
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 2,
            "channels": (8, 16),
            "strides": (2,),
        },
        (2, 1, 32, 32),
    ],
    # DenseNet
    [
        DenseNet121,
        {"spatial_dims": 2, "in_channels": 1, "out_channels": 2},
        (2, 1, 64, 64),
    ],
    # EfficientNet
    [
        EfficientNetBN,
        {"model_name": "efficientnet-b0", "spatial_dims": 2, "in_channels": 1, "num_classes": 2},
        (2, 1, 32, 32),
    ],
    # SENet
    [
        SENet154,
        {"spatial_dims": 2, "in_channels": 1, "num_classes": 2},
        (2, 1, 64, 64),
    ],
    # FullyConnectedNet
    [
        FullyConnectedNet,
        {"in_channels": 10, "out_channels": 2, "hidden_channels": [16, 16]},
        (2, 10),
    ],
    # AHNet
    [
        AHNet,
        {"spatial_dims": 2, "in_channels": 1, "out_channels": 2, "psp_block_num": 2},
        (2, 1, 32, 32),
    ],
]

# Test different compile modes
COMPILE_MODES = [None, "default", "reduce-overhead", "max-autotune"]


@unittest.skipUnless(has_torch_compile, "torch.compile not available (requires PyTorch 2.0+)")
class TestTorchCompileCompatibility(unittest.TestCase):
    """Test that MONAI networks are compatible with torch.compile."""

    @parameterized.expand(TEST_CASES)
    def test_network_compile(self, network_class, init_kwargs, input_shape):
        """Test that a network can be compiled and produces the same output."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        model = network_class(**init_kwargs).to(device)
        model.eval()

        # Create input
        x = torch.randn(input_shape, device=device)

        # Get output from original model
        with torch.no_grad():
            output_original = model(x)

        # Compile model
        compiled_model = torch.compile(model)

        # Get output from compiled model
        with torch.no_grad():
            output_compiled = compiled_model(x)

        # Check outputs match
        torch.testing.assert_close(
            output_compiled,
            output_original,
            rtol=1e-4,
            atol=1e-4,
            msg=f"{network_class.__name__} compiled output differs from original",
        )

    @parameterized.expand(
        [
            [network_class, init_kwargs, input_shape, mode]
            for network_class, init_kwargs, input_shape in TEST_CASES
            for mode in COMPILE_MODES
        ]
    )
    def test_network_compile_modes(self, network_class, init_kwargs, input_shape, mode):
        """Test that networks work with different compile modes."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create and compile model
        model = network_class(**init_kwargs).to(device)
        model.eval()

        if mode is not None:
            compiled_model = torch.compile(model, mode=mode)
        else:
            compiled_model = torch.compile(model)

        # Create input
        x = torch.randn(input_shape, device=device)

        # Test forward pass
        with torch.no_grad():
            output = compiled_model(x)

        # Basic shape check
        self.assertIsNotNone(output, f"{network_class.__name__} with mode={mode} returned None")
        self.assertIsInstance(output, torch.Tensor, f"{network_class.__name__} with mode={mode} did not return a Tensor")

    def test_compile_training_mode(self):
        """Test that compiled models work in training mode."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use a simple network for training test
        model = BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            features=(8, 8, 16, 16, 32, 32),
        ).to(device)

        # Compile model
        compiled_model = torch.compile(model)
        compiled_model.train()

        # Create dummy data
        x = torch.randn(2, 1, 32, 32, device=device)
        target = torch.randint(0, 2, (2, 32, 32), device=device)

        # Forward pass
        output = compiled_model(x)

        # Backward pass
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that gradients were computed
        has_grad = any(p.grad is not None for p in compiled_model.parameters() if p.requires_grad)
        self.assertTrue(has_grad, "No gradients computed in training mode")

    def test_compile_with_eval_mode(self):
        """Test that compiled models work with MONAI's eval_mode context manager."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            features=(8, 8, 16, 16, 32, 32),
        ).to(device)

        compiled_model = torch.compile(model)

        x = torch.randn(2, 1, 32, 32, device=device)

        # Test with eval_mode
        with eval_mode(compiled_model):
            output = compiled_model(x)

        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
