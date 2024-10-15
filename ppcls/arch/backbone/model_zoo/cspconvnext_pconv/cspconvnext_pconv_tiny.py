import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import TruncatedNormal, Constant

from .....utils.save_load import load_dygraph_pretrain


MODEL_URLS = {
    "ConvNeXt_tiny":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_tiny_pretrained.pdparams",
    "ConvNeXt_small":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_small_pretrained.pdparams",
    "ConvNeXt_base_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_base_224_pretrained.pdparams",
    "ConvNeXt_base_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_base_384_pretrained.pdparams",
    "ConvNeXt_large_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_large_224_pretrained.pdparams",
    "ConvNeXt_large_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_large_384_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


class Identity(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    

class EffectiveSELayer(nn.Layer):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2D(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

class Partial_conv3(nn.Layer):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2D(self.dim_conv3, self.dim_conv3, 3, 1, 1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self,x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self,x):
        # for training/inference
        x1, x2 = paddle.split(x, [self.dim_conv3, self.dim_untouched], axis=1)
        x1 = self.partial_conv3(x1)
        x = paddle.concat((x1, x2), 1)

        return x
    
class Block(nn.Layer):
    def __init__(self, dim, kernel_size=7, if_gourp=1, forward="split_cat", drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        if if_gourp == 1:
            groups = dim
        else:
            groups = 1
        self.dwconv = Partial_conv3(dim, 4, forward)
        self.norm = nn.BatchNorm2D(dim)
        self.pwconv1 = nn.Conv2D(dim, 
                                 4 * dim, 
                                 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2D(4 * dim, dim, 1)
        self.ese = EffectiveSELayer(dim, dim)
        self.norm2 = nn.BatchNorm2D(dim)
        self.gamma =  paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(
                value=1.0)
        ) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm2(x)
        x = self.ese(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = input + x
        return x
    

class L2Decay(paddle.regularizer.L2Decay):
    def __init__(self, coeff=0.0):
        super(L2Decay, self).__init__(coeff)


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups)

        self.bn = nn.BatchNorm2D(
            ch_out,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x
    
class CSPStage(nn.Layer):
    def __init__(self,
                block_fn,
                ch_in,
                ch_out,
                n,
                stride,
                p_rates,
                kernel_size=7,
                if_group=1,
                layer_scale_init_value=1e-6,
                act=nn.GELU,
                attn='eca',
                forward='split_cat'):
        super().__init__()
        ch_mid = (ch_in+ch_out)//2
        if stride == 2:
            self.down = nn.Sequential(ConvBNLayer(ch_in, ch_mid , 2, stride=2,  act=act))
        else:
            self.down = Identity()
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2,kernel_size, if_group, forward=forward,drop_path=p_rates[i],layer_scale_init_value=layer_scale_init_value)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = paddle.concat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

class CSPConvNext_Pconv(nn.Layer):
    def __init__(
        self,
        class_num=1000,
        in_chans=3,
        forward="split_cat",
        depths=[3, 3, 9, 3],
        dims=[64,128,256,512,1024],
        kernel_size=7,
        if_group=1,
        drop_path_rate=0.2,
        layer_scale_init_value=1e-6,
        stride=[2,2,2,2],
        return_idx=[1,2,3],
        depth_mult = 1.0,
        width_mult = 1.0,
        stem = "vb"
    ):
        super().__init__()
        block_former = [Block,Block,Block,Block]
        act = nn.GELU()

        if stem == "va":
            self.Down_Conv = nn.Sequential(
                ('conv1', ConvBNLayer(
                    in_chans,(dims[0]+dims[1])//2 , 4, stride=4,  act=act)),
            )
        if stem == "vb":
            self.Down_Conv = nn.Sequential(
                ('conv1', ConvBNLayer(
                    in_chans, dims[0]//2 , 2, stride=2,  act=act)),
                ('conv2', ConvBNLayer(
                    dims[0]//2, dims[0]//2 , 3, stride=1,padding=1,  act=act)),
                ('conv3', ConvBNLayer(
                    dims[0]//2, dims[0] , 3, stride=1,padding=1, act=act)),
            )
        
        dp_rates = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]
        n = len(depths)
        self.stages = nn.Sequential(*[(str(i), CSPStage(
            block_former[i], 
            dims[i], 
            dims[i + 1], 
            depths[i], 
            stride[i],
            dp_rates[sum(depths[:i]) : sum(depths[:i+1])],
            kernel_size=kernel_size,
            if_group=if_group,
            act=nn.GELU,
            forward=forward))
                                      for i in range(n)])

        self.norm = nn.LayerNorm(dims[-1], epsilon=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], class_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            try:
                trunc_normal_(m.weight)
                zeros_(m.bias)
            except:
                print(m)

    def forward_body(self, inputs):
        x = inputs
        x = self.Down_Conv(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
        return self.norm(x.mean([-2, -1]))
    
    def forward(self, x):
        x = self.forward_body(x)
        x = self.head(x)
        return x

def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def CSPConvNeXt_pconv_tiny(pretrained=False, use_ssld=False, **kwargs):
    model = CSPConvNext_Pconv(**kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXt_tiny"], use_ssld=use_ssld)
    return model

if __name__=="__main__":
     model  = CSPConvNeXt_pconv_tiny()
     # Total Flops: 1189500624     Total Params: 8688640
     paddle.flops(model,(1,3,224,224),print_detail=True)