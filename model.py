import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        # self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W


class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]
    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim,
                                       head_count=head_count)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
        self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # dual attention structure, efficient attention first then transpose attention
        norm1 = self.norm1(x)
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)
        attn = self.attn(norm1)
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)

        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)

        mx = add3 + mlp2
        return mx


# Encoder
class Encoder(nn.Module):
    def __init__(self, in_dim, layers):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        size = [(56, 56), (28, 28), (14, 14)]
        patch_dim1 = []
        in_s = [1] + in_dim[:-1]
        for i in range(len(in_dim)):
            patch_dim1.append(OverlapPatchEmbeddings(
                patch_sizes[i], strides[i], padding_sizes[i], in_s[i], in_dim[i]
            ))
        self.patch_embed_dim1 = nn.ModuleList(patch_dim1)

        # transformer encoder
        self.block_list = nn.ModuleList()
        self.norm_list = nn.ModuleList()
        for i in range(len(in_dim)):
            # dim, size, more_dim=True, more_tk=True
            # in_dim, key_dim, value_dim, head_count=1, token_mlp="mix_skip"
            mdl = nn.ModuleList(
                [DualTransformerBlock(in_dim[i], in_dim[i], in_dim[i]) for _ in range(layers[i])]
            )
            self.block_list.append(mdl)
            self.norm_list.append(nn.LayerNorm(in_dim[i]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B = x.shape[0]
        outs = []

        for i, blks in enumerate(self.block_list):
            x, h, w = self.patch_embed_dim1[i](x)
            for blk in blks:
                x = blk(x, h, w)
            x = self.norm_list[i](x)
            outs.append(x)
            x = Rearrange("b (h w) c -> b c h w", h=h, w=w)(x)

        return outs


# Decoder
class Patch_Expand(nn.Module):
    def __init__(self, size, out_size, dim, out_dim):
        super().__init__()
        # self.size = size
        self.linear = nn.Linear(dim, out_dim, bias=False)
        self.linear2 = nn.Conv1d(size[0] * size[1], out_size[0] * out_size[1], 1, bias=False)
        # self.linear3 = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = self.linear(x)
        x = self.linear2(x)
        # x = self.linear3(x)
        x = self.norm(x.clone())

        return x


class Final_Patch(nn.Module):
    def __init__(self, size, dim, dim_scale=4):
        super().__init__()
        self.size = size
        self.dim = dim
        self.dim_scale = dim_scale

        self.expand = nn.Linear(dim, dim_scale ** 2 * dim, bias=False)
        self.output_dim = dim
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.size
        x = self.expand(x)
        # x = Rearrange('b (h w) (p e) -> b (p h w) e', p=self.dim_scale ** 2 )(atten)

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, -1, self.output_dim)  # b, dim_scale**2 * hw, c
        x = self.norm(x.clone())

        return x  # b, dim_scale**2 * hw, c



class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    # hw不同,e相同
    def forward(self, x, x2):
        # shape: b, hw, e, 
        value = x
        key = F.softmax(x2.transpose(1, 2), dim=1)
        query = F.softmax(x2, dim=1)

        context = value @ key
        atten = context @ query

        atten = self.projection(atten)

        return atten


class SAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, x, x2):
        query = x
        key = x2.transpose(1, 2)
        value = x2

        context = F.softmax(query @ key, dim=2)
        atten = context @ value
        atten = self.projection(atten)

        return atten



class Feature_Fusion_Block(nn.Module):
    def __init__(self, dims, location, size, is_add=True, is_sa=False, more_2=True):

        super().__init__()
        self.dims = dims
        self.location = location
        self.size = size
        self.is_add = is_add
        self.more_2 = more_2
        self.attention = CrossAttention(dims[location])
        self.attention2 = CrossAttention(dims[location])
        if location == 0:
            self.linear = nn.Linear(dims[1], dims[0], bias=False)
            self.linear2 = nn.Linear(dims[2], dims[0], bias=False)
            self.linear3 = nn.Linear(dims[2], dims[1], bias=False)
            self.linear4 = nn.Linear(dims[1], dims[0], bias=False)

        elif location == 1:
            self.linear = nn.Linear(dims[0], dims[1], bias=False)
            self.linear2 = nn.Linear(dims[2], dims[1], bias=False)
            if more_2:
                self.linear1_0 = nn.Linear(dims[1], dims[0], bias=False)
                self.linear1_2 = nn.Linear(dims[1], dims[2], bias=False)
                self.linear0_1 = nn.Linear(dims[0], dims[1], bias=False)
                self.linear2_1 = nn.Linear(dims[2], dims[1], bias=False)
            if is_add:
                self.w2 = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
            else:
                self.linear3 = nn.Linear(dims[1] * 2, dims[1], bias=False)

        elif location == 2:
            self.linear = nn.Linear(dims[0], dims[2], bias=False)
            self.linear2 = nn.Linear(dims[1], dims[2], bias=False)
            self.linear3 = nn.Linear(dims[0], dims[1], bias=False)
            self.linear4 = nn.Linear(dims[1], dims[2], bias=False)

            self.w = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        else:
            self.linear5 = nn.Linear(dims[location] * 3, dims[location], bias=False)

        if location != 1:
            self.attention3 = CrossAttention(dims[1])
            self.attention4 = CrossAttention(dims[location])
        elif more_2:

            if is_sa:
                self.attention3 = SAttention(dims[1])
                self.attention4 = SAttention(dims[1])
            else:
                self.attention3 = CrossAttention(dims[0])
                self.attention4 = CrossAttention(dims[2])

        self.norm = nn.LayerNorm(dims[location])
        self.mlp = MixFFN_skip(dims[location], dims[location] * 4)

    # 56*56,128 28*28,320/256 14*14,512
    def forward(self, x, x2, x3):
        if self.location == 0:
            image = x
            image2 = self.linear(x2)
            image3 = self.linear2(x3)
            image4 = self.linear3(x3)

            atten = x
            atten1 = self.attention(image, image2)
            atten2 = self.attention2(image, image3)
            # atten3 = self.projection(x2)
            atten3 = self.attention3(x2, image4)
            atten3 = self.linear4(atten3)
            atten3 = self.attention4(image, atten3)
            # print(atten3.shape)

        elif self.location == 1:
            image = self.linear(x)
            image2 = x2
            image3 = self.linear2(x3)
            atten = x2
            atten1 = self.attention(image2, image)
            atten2 = self.attention2(image2, image3)
            if self.more_2:
                
                atten3 = self.linear1_0(image2)
                atten4 = self.linear1_2(image2)
                atten3 = self.attention3(atten3, x)
                atten3 = self.linear0_1(atten3)
                atten4 = self.attention4(atten4, x3)
                atten4 = self.linear2_1(atten4)
            if self.is_add:
                f2 = self.w2.sigmoid()
                atten3 = f2.expand_as(atten3) * atten3 + (1 - f2).expand_as(atten4) * atten4
            else:
                atten3 = torch.cat((atten3, atten4), dim=2)
                atten3 = self.linear3(atten3)

        elif self.location == 2:
            image = self.linear(x)
            image2 = self.linear2(x2)
            image3 = x3
            image4 = self.linear3(x)
            atten = x3
            atten1 = self.attention(image3, image)
            atten2 = self.attention2(image3, image2)
            # atten3 = self.projection(x2)
            atten3 = self.attention3(x2, image4)
            atten3 = self.linear4(atten3)
            atten3 = self.attention4(image3, atten3)
            # print(atten3.shape)

        if self.is_add:
            f = self.w.sigmoid()
            other = (1 - f) / 2
            atten_ = f.expand_as(atten3) * atten3 + other.expand_as(atten2) * atten2 + other.expand_as(atten1) * atten1
        else:
            atten_ = torch.cat((atten1, atten2, atten3), dim=2)
            atten_ = self.linear5(atten_)

        atten = atten + atten_


        atten_norm = self.norm(atten)
        atten = atten + self.mlp(atten_norm, *self.size)
        return atten


class Cross_Attention(nn.Module):

    def __init__(self, dim, is_add=True):
        super().__init__()

        self.is_add = is_add
        self.attention = CrossAttention(dim)
        self.attention2 = CrossAttention(dim)

        if is_add:
            self.w = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        else:
            self.linear = nn.Linear(dim * 2, dim, bias=False)
        # self.linear3 = nn.Linear(dim*2, dim)

    # x2,x3-> image_low, image_up
    # shape: b, hw, e
    def forward(self, x, x2, x3=None):

        atten = self.attention(x, x2)

        if x3 is not None:
            atten2 = self.attention2(x, x3)

            if self.is_add:
                f = self.w.sigmoid()
                atten = atten * f.expand_as(atten) + atten2 * f.expand_as(atten2)
            else:
                atten3 = torch.cat((atten, atten2), dim=2)
                atten = self.linear(atten3)


        atten = atten + x

        return atten


class Mering_Attention_Block(nn.Module):
    def __init__(self, in_size, out_size, in_dim, out_dim):
        super().__init__()

        self.in_size = in_size

        self.mix_ffn = MixFFN_skip(in_dim, in_dim * 4)
        self.norm = nn.LayerNorm(in_dim)
        # self.norm2 = nn.LayerNorm(out_dim)
        self.attention = Cross_Attention(in_dim)

        self.linear = nn.Linear(out_dim, in_dim, bias=False)
        self.patch_expand = Patch_Expand(in_size, out_size, in_dim, out_dim)
        # self.patch_expand2 = Patch_Expand(in_size, out_size, in_dim, out_dim)
    # shape: b, hw, c
    def forward(self, image, image_up, image_low):
        if len(image.shape) == 4:  # b,embeding, h,w
            image = Rearrange('b e h w -> b (h w) e ')(image)



        image_up_ = self.linear(image_up) 

        # attention
        atten = self.attention(image, image_low, image_up_)
        atten_norm = self.norm(atten)
        atten = self.mix_ffn(atten_norm, *self.in_size) + atten
        expand_ = self.patch_expand(atten)

        # expand_2 = self.patch_expand2(image)
        # atten = expand_ + expand_2
        atten = image_up + expand_ 
        # atten = self.norm2(atten)
        return atten


class Decoder_Block(nn.Module):
    def __init__(self, in_size, out_size, in_dim, out_dim, n_class=9):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.in_dim = in_dim
        # in_size, out_size, in_dim, out_dim
        self.attention = Mering_Attention_Block(in_size, out_size, in_dim, out_dim)
        self.layer_former_1 = DualTransformerBlock(out_dim, out_dim, out_dim, head_count=1)
        self.layer_former_2 = DualTransformerBlock(out_dim, out_dim, out_dim, head_count=1)
        self.norm = nn.LayerNorm(out_dim)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x, x2, x3=None):
        # image, image_up, image_low
        atten = self.attention(x, x2, x3)
        # atten = self.patch_expand(atten)
        atten = self.layer_former_1(atten, *self.out_size)
        atten = self.layer_former_2(atten, *self.out_size)
        atten = self.norm(atten)

        return atten


class MyFormer(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        dims, layers = [128, 320, 512], [2, 2, 2]

        self.encoder = Encoder(
            in_dim=dims,
            layers=layers
        )
        base_size = 7
        self.bs = 7
        self.fusion_block = nn.ModuleList(
            [Feature_Fusion_Block(dims, i, (base_size * 2 ** (3 - i), base_size * 2 ** (3 - i))) for i in
             range(len(dims))]
        )
        self.layer_former_1 = DualTransformerBlock(dims[2], dims[2], dims[2], head_count=1)
        self.layer_former_2 = DualTransformerBlock(dims[2], dims[2], dims[2], head_count=1)
       
        self.final_patch = Final_Patch(size=(base_size * 8, base_size * 8), dim=dims[0], dim_scale=4)
        self.last_num_class = nn.Conv2d(dims[0], num_classes, 1)
        # self.last_num_class2 = nn.Conv2d(dims[0], num_classes, 1)
        self.decoder_2 = Decoder_Block(
            in_size=(base_size * 2, base_size * 2),
            out_size=(base_size * 4, base_size * 4),
            in_dim=dims[2],
            out_dim=dims[1],
            n_class=num_classes,
        )
        self.decoder_1 = Decoder_Block(
            in_size=(base_size * 4, base_size * 4),
            out_size=(base_size * 8, base_size * 8),
            in_dim=dims[1],
            out_dim=dims[0],
            n_class=num_classes,
        )


    def forward(self, x):
        output = self.encoder(x)
        fusion = [self.fusion_block[i](*output) for i in range(len(output))]

        decoder3 = fusion[2]

        decoder3 = self.layer_former_1(decoder3, self.bs * 2, self.bs * 2)
        decoder3 = self.layer_former_2(decoder3, self.bs * 2, self.bs * 2)

        decoder2 = self.decoder_2(decoder3, fusion[1], fusion[2])  # 28*28, 320
        decoder1 = self.decoder_1(decoder2, fusion[0], fusion[1])  # 56*56, 128
       

        image = Rearrange('b (h w) c -> b c h w', h=224, w=224)(self.final_patch(decoder1))

        image = self.last_num_class(image)


        return image




if __name__ == "__main__":
    data = torch.Tensor(1, 1, 224, 224).cuda()
    model = MyFormer().cuda()
    output = model(data)
    print(f"output shape: {output.shape}")
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .4fM' % (total / 1e6))
    from thop import profile
    Flops, params = profile(model, inputs=(data,))  
    print('Flops: % .4fG' % (Flops / 1000000000))  
    print('params: % .4fM' % (params / 1000000))  
