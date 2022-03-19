import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Union, List
from models.transformer import PositionalEncoding, EncoderLayer
from models.sem_graph_conv import SemGraphConv


class ErrorRefinement(nn.Module):
    """
    The error refinement network that refines a predicted 3D pose using the initial pose and the predicted error.

    Arguments
    ---------
    num_joints_in: int
        Number of joints per pose. Default=16 for human36m.
    n_features: Tuple[int]
        Number of features per joint. Tuple for input and output features. Default=(2, 3) for 2D->3D poses.
    n_layer: int
        Number of transformer encoder layers. Default=4.
    d_model: int
        Size of the hidden dimension per transformer encoder. Default=128.
    d_inner: int
        Size of the hidden dimension within the feed forward module inside the encoder transformer layers. Default=512.
    n_head: int
        Number of multi-head attention head. Default=8.
    d_k: int
        Size of the keys within the multi-head attention modules. Default=64.
    d_v: int
        Size of the values within the multi-head attention modules. Default=64.
    encoder_dropout: float
        Dropout probability within the transformer encoder layers. Default=0.0.
    pred_dropout: float
        Dropout probability for the prediction layers. Default=0.2.
    intermediate: bool
        Set to True for intermediate supervision. In this case the output pose is predicted
        after each transformer encoder layer and returned in a list. Default=False.
    spatial_encoding: bool
        Set to True for spatial encoding of the input poses instead of the default positional encoding. Default=False.

    Methods
    -------
    forward(torch.Tensor): Union[List[torch.Tensor], torch.Tensor]
        Forward pass through the module. Given an input pose, predicted pose and predicted error will refine the predicted pose.
    """

    def __init__(self, num_joints_in: int=16, n_features: Tuple[int]=(2, 3), n_layers: int=4, d_model: int=128, 
                 d_inner: int=512, n_head: int=8, d_k: int=64, d_v: int=64, encoder_dropout: float=0.0, pred_dropout: float=0.2,
                 intermediate: bool=False, spatial_encoding: bool=False) -> None:

        """
        Initialize the network.
        """

        super(ErrorRefinement, self).__init__()

        self.spatial_encoding = spatial_encoding
        self.intermediate = intermediate

        self.embedding = nn.Conv1d(n_features[0] + 2 * n_features[1], d_model, 1)  # Input features + predicted features + predicted error -> hidden dimension.
        self.embedding_forward = lambda x: self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Positional encoding.
        if self.spatial_encoding:
            self.position_enc = nn.Parameter(torch.zeros(1, num_joints_in, d_model))    
        self.dropout = nn.Dropout(p=encoder_dropout)

        # Stacked encoders.
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=encoder_dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output prediction.
        if self.intermediate:
            self.intermediate_pred = nn.ModuleList([nn.Sequential(nn.LayerNorm(num_joints_in * d_model), nn.Dropout(p=pred_dropout), nn.Linear(num_joints_in * d_model, num_joints_in * n_features[1])) for _ in range(n_layers)])
            self.intermediate_enc = nn.ModuleList([nn.Linear(num_joints_in * n_features[1], num_joints_in * d_model) for _ in range(n_layers)])
        else:
            self.decoder = nn.Sequential(
                nn.LayerNorm(num_joints_in * d_model),
                nn.Dropout(p=pred_dropout),                        
                nn.Linear(num_joints_in * d_model, num_joints_in * n_features[1])
            )

        # Initialize layers with xavier.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def forward(self, src: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the network. Input is a sequence of 2D joints, predicted 3D joints and predicted error belonging to one pose.
        
        Parameters
        ----------
        src: torch.Tensor
            Tensor of 2D joints, predicted 3D joints and predicted error. [B, J, 2+3+3], where J is the number of joints.

        Returns
        -------
        out: Union[List[torch.Tensor], torch.Tensor]
            The predicted 3D pose for each joint. If intermediate is true, this is a list of predicted 3D poses for each encoder layer.
            The shape is [B, J, 3]. In case a list is returned, they are ordered from first encoder layer to last encoder layer.
        """

        b, j, _ = src.shape  # Batch, Number of joints, Number of features per joint
        intermediate_list = []

        # Expand dimensions.
        src = self.embedding_forward(src)
        
        # Positional encoding.
        if self.spatial_encoding:
            src += self.position_enc
 
        enc_output = self.dropout(src)
        enc_output = self.layer_norm(enc_output)

        # Stack of encoders.
        for i, enc_layer in enumerate(self.layer_stack):
            residual = enc_output
            enc_output, _ = enc_layer(enc_output, slf_attn_mask=None)

            if self.intermediate:
                pred = self.intermediate_pred[i](enc_output.clone().view(b, -1))
                res = self.intermediate_enc[i](pred).view(b, j, -1)

                enc_output += res
                intermediate_list += [pred]

            enc_output += residual
            enc_output = F.relu(enc_output)
 
        # Final prediction.
        if self.intermediate:
            out = [out.view(b, j, -1) for out in intermediate_list]
        else:  
            out = self.decoder(enc_output.view(b, -1)).view(b, j, -1)

        return out


class JointTransformer(nn.Module):
    """
    The joint transformer model that predicts 3D poses from 2D poses. 

    Arguments
    ---------
    num_joints_in: int
        Number of joints per pose. Default=16 for human36m.
    n_features: Tuple[int]
        Number of features per joint. Tuple for input and output features. Default=(2, 3) for 2D->3D poses.
    n_layer: int
        Number of transformer encoder layers. Default=4.
    d_model: int
        Size of the hidden dimension per transformer encoder. Default=128.
    d_inner: int
        Size of the hidden dimension within the feed forward module inside the encoder transformer layers. Default=512.
    n_head: int
        Number of multi-head attention head. Default=8.
    d_k: int
        Size of the keys within the multi-head attention modules. Default=64.
    d_v: int
        Size of the values within the multi-head attention modules. Default=64.
    encoder_dropout: float
        Dropout probability within the transformer encoder layers. Default=0.0.
    pred_dropout: float
        Dropout probability for the prediction layers. Default=0.2.
    intermediate: bool
        Set to True for intermediate supervision. In this case the output pose is predicted
        after each transformer encoder layer and returned in a list. Default=False.
    spatial_encoding: bool
        Set to True for spatial encoding of the input poses instead of the default positional encoding. Default=False.
    embedding_type: str
        Type of layer to use to embed the input coordinates to the hidden dimension. Default='conv'.
    adj: torch.Tensor
        Adjacency matrix for the skeleton. Only needed if embedding_type='graph'. Default=None.
    error_prediction: bool
        Set to True to predict the error in the output prediction for each joint. Default=True.

    Methods
    -------
    forward(torch.Tensor, torch.Tensor, bool): Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor, Union[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]
        Forward pass through the module. Given an input pose and optional image features, will predict the output pose.
        Returns the predicted pose, the last hidden state of the transformer encoders and the predicted error.
    """

    def __init__(self, num_joints_in: int=16, n_features: Tuple[int]=(2, 3), n_layers: int=4, d_model: int=128, 
                 d_inner: int=512, n_head: int=8, d_k: int=64, d_v: int=64, encoder_dropout: float=0.0, pred_dropout: float=0.2, 
                 intermediate: bool=False, spatial_encoding: bool=False, embedding_type: str='conv', adj: torch.Tensor=None,
                 error_prediction: bool=True) -> None:

        """
        Initialize the network.
        """

        super(JointTransformer, self).__init__()

        self.intermediate = intermediate
        self.error_prediction = error_prediction
        self.spatial_encoding = spatial_encoding
        assert embedding_type in ['conv', 'linear', 'graph'], 'The chosen embedding type \'{}\' is not supported.'.format(embedding_type)
        self.embedding_type = embedding_type

        # Expand from x,y input coordinates to the size of the hidden dimension.
        if embedding_type == 'conv':
            self.embedding = nn.Conv1d(n_features[0], d_model, 1)
            self.embedding_forward = lambda x: self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif embedding_type == 'linear':
            self.embedding = nn.Linear(n_features[0] * num_joints_in, d_model * num_joints_in)
            self.embedding_forward = lambda x: self.embedding(x.reshape(x.size(0), -1)).view(x.size(0), x.size(1), -1)
        elif embedding_type == 'graph':
            self.embedding = SemGraphConv(n_features[0], d_model, adj)
            self.embedding_forward = lambda x: self.embedding(x)
        else:
            raise NotImplementedError
        
        # Positional encoding.
        if self.spatial_encoding:
            self.position_enc = nn.Parameter(torch.zeros(1, num_joints_in, d_model)) 
        self.dropout = nn.Dropout(p=encoder_dropout)

        # Stacked encoders.
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=encoder_dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output prediction layers.
        if self.intermediate:
            self.intermediate_pred = nn.ModuleList([nn.Sequential(nn.LayerNorm(num_joints_in * d_model), nn.Dropout(p=pred_dropout), nn.Linear(num_joints_in * d_model, num_joints_in * n_features[1])) for _ in range(n_layers)])
            self.intermediate_enc = nn.ModuleList([nn.Linear(num_joints_in * n_features[1], num_joints_in * d_model) for _ in range(n_layers)])
            if self.error_prediction:
                self.intermediate_error = nn.ModuleList([nn.Sequential(nn.LayerNorm(num_joints_in * d_model), nn.Dropout(p=pred_dropout), nn.Linear(num_joints_in * d_model, num_joints_in * n_features[1])) for _ in range(n_layers)])
        else:
            self.decoder = nn.Sequential(
                nn.LayerNorm(num_joints_in * d_model),
                nn.Dropout(p=pred_dropout),                        
                nn.Linear(num_joints_in * d_model, num_joints_in * n_features[1])
            )
            if self.error_prediction:
                self.error_decoder = nn.Sequential(
                    nn.LayerNorm(num_joints_in * d_model),
                    nn.Dropout(p=pred_dropout),
                    nn.Linear(num_joints_in * d_model, num_joints_in * n_features[1])
                )

        # Initialize layers with xavier.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def forward(self, src: torch.Tensor, image: torch.Tensor=None, return_attns: bool=False) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor, Union[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the network. Input is a sequence of 2D joints belonging to one pose.
        
        Parameters
        ----------
        src: torch.Tensor
            Tensor of 2D joints. [B, J, 2], where J is the number of joints.
        image: torch.Tensor
            Tensor of cropped image features around each joint. [B, J, H, W].
            This tensor is not currently used.
        return_attns: bool
            Set to True if the self attention tensors should be returned.

        Returns
        -------
        out: Union[List[torch.Tensor], torch.Tensor]
            The predicted 3D pose for each joint. If intermediate is true, this is a list of predicted 3D poses for each encoder layer.
            The shape is [B, J, 3]. In case a list is returned, they are ordered from first encoder layer to last encoder layer.
        enc_output: torch.Tensor
            The final hidden state that is the output of the last encoder layer. These features can be further used for sequence prediction.
            The shape is [B, J, d_model].
        error: Union[List[torch.Tensor], torch.Tensor]
            The predicted error of the 3D pose for each joint. If intermediate is true, this is a list of predicted errors for each encoder layer.
            The shape is [B, J, 3]. In case a list is returned, they are ordered from first encoder layer to last encoder layer.
        return_attns: List[torch.Tensor]
            Optional attention maps for every transformer encoder in the stack.
        """

        b, j, _ = src.shape  # Batch, Number of joints, Number of features per joint
        intermediate_list = []
        error_list = []
        enc_slf_attn_list = []

        # Expand dimensions.
        src = self.embedding_forward(src)
        
        # Positional encoding.
        if self.spatial_encoding:
            src += self.position_enc
 
        enc_output = self.dropout(src)
        enc_output = self.layer_norm(enc_output)

        # Stack of encoders.
        for i, enc_layer in enumerate(self.layer_stack):
            residual = enc_output
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=None)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

            if self.intermediate:
                pred = self.intermediate_pred[i](enc_output.clone().view(b, -1))
                res = self.intermediate_enc[i](pred).view(b, j, -1)
                if self.error_prediction:
                    error = self.intermediate_error[i](enc_output.clone().view(b, -1))
                    error_list += [error]

                enc_output += res
                intermediate_list += [pred]

            enc_output += residual
            enc_output = F.relu(enc_output)

        # Output either predictions for each encoder, or one prediction at the end. Always also output the last encoder output.
        if self.intermediate:
            out = [out.view(b, j, -1) for out in intermediate_list]
            error = [e.view(b, j, -1) for e in error_list]
        else:  
            out = self.decoder(enc_output.view(b, -1)).view(b, j, -1)
            error = self.error_decoder(enc_output.view(b, -1)).view(b, j, -1) if self.error_prediction else None

        if return_attns:
            return out, enc_output, error, enc_slf_attn_list
        else:
            return out, enc_output, error
