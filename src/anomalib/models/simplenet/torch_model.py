"""Pytorch model for the SimpleNet model implementation"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import timm
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import fsolve
from torch import Tensor, nn


class RunningAverage:
    """
    A class to calculate the running average of given values.
    """

    def __init__(self):
        self.count = 0
        self.avg_std = 0
        self.avg_mean = 0

    def update(self, mean, std):
        """
        Updates the running averages with new values.

        Args:
            mean (float): The new mean value to be included in the average.
            std (float): The new standard deviation value to be included in the average.

        Returns:
            tuple: A tuple containing the updated average mean and average standard deviation.
        """
        self.count += 1
        self.avg_std = self.avg_std + (std - self.avg_std) / self.count
        self.avg_mean = self.avg_mean + (mean - self.avg_mean) / self.count
        return self.avg_mean, self.avg_std


class FeatureExtractor:
    """
    A class to extract features from specific layers of a model.
    """

    def __init__(self, model_name, layer_list):
        """
        Initializes the FeatureExtractor with a model and selected layers.

        Args:
            model_name (str): Name of the model to load.
            layer_list (list): List of layer names to extract features from.
        """
        self.model = timm.create_model(model_name, pretrained=True, pretrained_cfg=None)
        self.model.eval()
        self.selected_layers = layer_list
        self.hooks = []
        for layer_name in self.selected_layers:
            layer = dict([*self.model.named_modules()])[layer_name]
            hook = layer.register_forward_hook(self._hook_fn)
            self.hooks.append(hook)

    def _hook_fn(self, module, input, output):
        self.features.append(output)

    def __call__(self, x):
        self.features = []
        device = x.device
        self.model.to(device)
        with torch.no_grad():
            _ = self.model(x)
        return self.features

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def get_feature_shape(self, shape):
        """
        Determines the shape of features for a hypothetical input.

        Args:
            shape (tuple): Shape of the hypothetical input tensor.

        Returns:
            list: Shapes of features from the selected layers.
        """
        self.features = []
        test_input = torch.rand(shape)
        with torch.no_grad():
            _ = self.model(test_input)
        shape_list = []
        for i in self.features:
            shape_list.append(i.shape[1::])
        return shape_list


class PatchMaker:
    """
    A class to create and handle patches from feature maps.
    """

    def __init__(self, patchsize, top_k=0, stride=None):
        """
        Initializes the PatchMaker with given patch size, top_k, and stride.

        Args:
            patchsize (int): Size of each patch.
            top_k (int, optional): The number of top elements to consider in scoring. Defaults to 0.
            stride (int, optional): Stride for patch extraction. Defaults to None.
        """
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """
        Converts a feature map tensor into respective patches.

        Args:
            features (torch.Tensor): The input tensor of shape [bs x c x w x h].
            return_spatial_info (bool, optional):
            If True, returns spatial information along with patches.
            Defaults to False.

        Returns:
            torch.Tensor: Tensor of patches or tuple of tensor of patches and spatial information.
            [torch.Tensor, bs * w//stride * h//stride, c, patchsize, patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        """
        Reshapes the scored patches back to the original batch size.

        Args:
            x (torch.Tensor): Tensor of patches.
            batchsize (int): The batch size to reshape to.

        Returns:
            torch.Tensor: Reshaped tensor.
        """
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        """
        Scores the input tensor, reducing its dimensions.

        Args:
            x (torch.Tensor or np.ndarray): The input tensor or numpy array.

        Returns:
            torch.Tensor or np.ndarray: Scored tensor or numpy array.
        """
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x


## ??????????????????
class MeanMapper(torch.nn.Module):
    """
    A PyTorch module for mapping features to a specified dimension using average pooling.
    """

    def __init__(self, preprocessing_dim):
        """
        Initializes the MeanMapper module with a specified output dimension.

        Args:
            preprocessing_dim (int): The target dimension for the output features.
        """
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        """
        Processes the input features using adaptive average pooling.

        Args:
            features (torch.Tensor): The input tensor of features.

        Returns:
            torch.Tensor: The processed tensor with features mapped to the specified dimension.
        """
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(torch.nn.Module):
    """
    A PyTorch module for preprocessing a set of features.
    """

    def __init__(self, input_dims, output_dim):
        """
        Initializes the Preprocessing module with specified input and output dimensions.

        Args:
            input_dims (list): A list of dimensions for each input feature.
            output_dim (int): The dimension of the output after processing.
        """
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        """
        Processes each feature in the input list using the corresponding module.

        Args:
            features (list): A list of input features to be processed.

        Returns:
            torch.Tensor: A tensor of processed features, stacked along the second dimension.
        """
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class Aggregator(torch.nn.Module):
    """
    A PyTorch module for aggregating features into a target dimension.
    """

    def __init__(self, target_dim):
        """
        Initializes the Aggregator module with a specified target dimension.

        Args:
            target_dim (int): The target dimension for the output features.
        """
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """
        Processes the input features to aggregate them into the target dimension.

        Args:
            features (torch.Tensor): The input tensor of features.

        Returns:
            torch.Tensor: The processed tensor with features aggregated to the target dimension.
        """
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class Projection(torch.nn.Module):
    """
    A PyTorch module that creates a sequence of linear layers for projection.
    """

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        """
        Initializes the Projection module.

        Args:
            in_planes (int): The dimensionality of the input features.
            out_planes (int, optional): The dimensionality of the output features. Defaults to in_planes.
            n_layers (int, optional): The number of linear layers to use. Defaults to 1.
            layer_type (int, optional): Determines the activation layers used. Defaults to 0.
        """
        super().__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(0.2))

    def forward(self, x):
        """
        Defines the forward pass of the Projection module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after projection.
        """
        x = self.layers(x)
        return x


class Discriminator(torch.nn.Module):
    """
    A Discriminator module.

    Attributes:
        body (torch.nn.Sequential): Sequential container of the main discriminator layers.
        tail (torch.nn.Linear): Final linear layer of the discriminator.
    """

    def __init__(self, in_planes, n_layers=1, hidden=None):
        """
        Initializes the Discriminator module.

        Args:
            in_planes (int): Dimensionality of the input features.
            n_layers (int, optional): Number of layers in the discriminator.
                                      Defaults to 1.
            hidden (int, optional): Dimensionality of the hidden layers.
                                    If None, it's set dynamically. Defaults to None.
        """
        super().__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module(
                "block%d" % (i + 1),
                torch.nn.Sequential(
                    torch.nn.Linear(_in, _hidden), torch.nn.BatchNorm1d(_hidden), torch.nn.LeakyReLU(0.2)
                ),
            )
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)

    def forward(self, x):
        """
        Defines the forward pass of the Discriminator.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor representing the discriminator's decision.
        """
        x = self.body(x)
        x = self.tail(x)
        return x


class RescaleSegmentor:
    """
    A class for rescaling segmentation scores and features to a target size.

    Attributes:
        target_size (int, tuple): The target size for rescaling.
        smoothing (int): The sigma value for Gaussian smoothing.
    """

    def __init__(self, target_size=224):
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores, features):
        """
        Rescales patch scores and features to the target size and applies Gaussian smoothing to the scores.

        Args:
            patch_scores (np.ndarray or torch.Tensor): The patch scores to be rescaled.
            features (np.ndarray or torch.Tensor): The features to be rescaled.

        Returns:
            tuple: A tuple containing smoothed patch scores and rescaled features.
        """
        patch_scores = np.expand_dims(patch_scores, axis=1)
        patch_scores = self._rescale(patch_scores)
        patch_scores = np.squeeze(patch_scores, axis=1)
        features = self._rescale(features, permute=True)

        smoothed_scores = [ndimage.gaussian_filter(score, sigma=self.smoothing) for score in patch_scores]
        return smoothed_scores, features

    def _rescale(self, data, permute=False):
        """
        Helper method to rescale data to the target size.

        Args:
            data (np.ndarray or torch.Tensor): The data to be rescaled.
            permute (bool, optional): Whether to permute the dimensions for feature data. Defaults to False.

        Returns:
            np.ndarray: The rescaled data.
        """
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if permute:
                data = data.permute(0, 3, 1, 2)
            data = F.interpolate(data, size=self.target_size, mode="bilinear", align_corners=False)
        return data


# the Bhattacharyya distance between two normal distributions
def equations(sigma2, D_B, mu1, sigma1, mu2):
    term1 = 1 / 4 * np.log(1 / 4 * ((sigma1**2 / sigma2**2) + (sigma2**2 / sigma1**2) + 2))
    term2 = 1 / 4 * ((mu1 - mu2) ** 2 / (sigma1**2 + sigma2**2))
    return term1 + term2 - D_B


class SimplenetModel(nn.Module):
    """
    Simplenet Module
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        patchsize: int,
        patchstride: int,
        layers: list[str],
        backbone: str = "wide_resnet50_2",
    ) -> None:
        super().__init__()
        print(input_size)
        self.layers = layers
        self.input_size = input_size
        self.patchsize = patchsize
        self.patchstride = patchstride

        self.backbone = backbone
        self.feature_extractor = FeatureExtractor(self.backbone, self.layers)

        feature_shape_list = self.feature_extractor.get_feature_shape((1, 3, input_size[0], input_size[1]))

        # current support 2 feature maps
        feature1_channel = feature_shape_list[0][0]  # 512
        feature2_channel = feature_shape_list[1][0]  # 1024

        self.patch_maker = PatchMaker(patchsize=patchsize, stride=patchstride)

        self.preprocessing = Preprocessing([feature1_channel, feature2_channel], feature1_channel + feature2_channel)

        self.target_embed_dimension = feature1_channel + feature2_channel
        self.preadapt_aggregator = Aggregator(target_dim=self.target_embed_dimension)

        self.pre_projection = Projection(
            in_planes=self.target_embed_dimension, out_planes=self.target_embed_dimension, n_layers=1, layer_type=0
        )

        self.discriminator = Discriminator(in_planes=self.target_embed_dimension, n_layers=2, hidden=feature2_channel)

        self.dsc_margin = 0.5

        self.mix_noise = 1
        self.noise_std = 0
        self.adp_noist_std = RunningAverage()

        self.anomaly_segmentor = RescaleSegmentor(target_size=input_size[-2:])

    def forward(self, batch: Tensor):
        batchsize = batch.shape[0]
        device = batch.device
        extracted_feature = self.feature_extractor(batch)
        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in extracted_feature]

        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]  # torch.Size([8, 324, 1024, 3, 3])
            patch_dims = patch_shapes[i]  # [18, 18]

            # Rearrange dimensions to get patches to the last dimensions, and flatten the patches
            _features = rearrange(_features, "b (p1 p2) c h w -> b c h w p1 p2", p1=patch_dims[0], p2=patch_dims[1])

            # Reshape the last two dimensions to combine all the patch dimensions into one, then interpolate
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )

            # Reshape back to original dimensions, and rearrange to get back to the desired ordering
            _features = _features.squeeze(1)
            _features = rearrange(
                _features,
                "(b c c1 c2) h w -> b c c1 c2 h w",
                b=perm_base_shape[0],
                c=perm_base_shape[1],
                c1=perm_base_shape[2],
                c2=perm_base_shape[3],
            )

            # Flatten the spatial dimensions
            _features = rearrange(_features, "b c c1 c2 h w -> b (h w) c c1 c2")
            features[i] = _features

        # Reshape the rest of the features
        features = [rearrange(x, "b q c c1 c2 -> (b q) c c1 c2") for x in features]
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.preprocessing(features)  # pooling each feature to same channel and stack together
        features = self.preadapt_aggregator(features)  # further pooling

        if self.training:
            true_feats = self.pre_projection(features)
            true_feats_std = torch.std(true_feats).detach().cpu().item()
            true_feats_mean = torch.mean(true_feats).detach().cpu().item()

            cur_mean, cur_std = self.adp_noist_std.update(true_feats_mean, true_feats_std)

            if self.adp_noist_std.count % 10 == 1:
                # Initial guess for sigma2
                initial_guess = cur_std / 5
                # Solve for sigma2
                sigma2_solution = fsolve(equations, initial_guess, args=(0.5, cur_mean, cur_std, 0))
                self.noise_std = sigma2_solution[0]

            noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
            noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise)
            noise = torch.stack(
                [torch.normal(0, self.noise_std * 1.1 ** (k), true_feats.shape) for k in range(self.mix_noise)], dim=1
            )
            noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
            noise = noise.to(device)
            fake_feats = true_feats + noise

            scores = self.discriminator(torch.cat([true_feats, fake_feats]))
            true_scores = scores[: len(true_feats)]
            fake_scores = scores[len(fake_feats) :]

            th = self.dsc_margin
            (true_scores.detach() >= th).sum() / len(true_scores)
            (fake_scores.detach() < -th).sum() / len(fake_scores)
            true_loss = torch.clip(-true_scores + th, min=0)
            fake_loss = torch.clip(fake_scores + th, min=0)

            loss = true_loss.mean() + fake_loss.mean()

            return loss
        else:
            with torch.no_grad():
                feats = self.pre_projection(features)

                patch_scores = image_scores = -self.discriminator(feats)
                patch_scores = patch_scores.cpu().numpy()
                # image_scores = image_scores.cpu().numpy()

                image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
                image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
                image_scores = self.patch_maker.score(image_scores)

                patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
                scales = patch_shapes[0]
                patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
                feats = feats.reshape(batchsize, scales[0], scales[1], -1)
                masks, feats = self.anomaly_segmentor.convert_to_segmentation(patch_scores, feats)
            tensor_masks = [torch.tensor(mask).unsqueeze(0) for mask in masks]
            masks_mat = torch.stack(tensor_masks, dim=0)
            masks_mat = masks_mat.to(device)

            return features, masks_mat, image_scores
