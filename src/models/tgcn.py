import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop

# Graph Convolutional Neural Network Module ###############################################################################
class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        """Constructor of TGCNGraphConvolution class for GCN module.

        Args:
            adj ([type]): adjacency matrix
            num_gru_unit (int): number of GRU Unit
            output_dim (int): output dimension
            bias (float, optional): bias of the GRU neural networks. Defaults to 0.0.
        """
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Resets hyperparameters, convert to tensor."""
        nn.init.xavier_uniform_(
            self.weights
        )  # fills the input tensor with values using a uniform distribution
        nn.init.constant_(
            self.biases, self._bias_init_value
        )  # fills the input self.biases with the value self._bias_init_values

    def forward(self, inputs, hidden_state):
        """Executes forward pass of the TGCNGraphConvolution class.

        Args:
            inputs ([type]): input features
            hidden_state ([type]): hidden state

        Returns:
            [type]: outputs
        """
        batch_size, num_nodes = inputs.shape

        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))

        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )

        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)

        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)

        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )

        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation

        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )

        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)

        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )

        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases

        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))

        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))

        return outputs

    @property
    def hyperparameters(self):
        """Sets hyperparameter values for GCN module.

        Returns:
            [type]: num_gru_units, output_dim, bias_init_value
        """
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


# Gated Reccurent Unit Module embedded two GCN Modules ###############################################################################
class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        """Constructor of TGCNCell class for GRU module. Each GRU module contain 2 GCN module.

        Args:
            adj ([type]): adjacency matrix
            input_dim (int): input dimension
            hidden_dim (int): hidden dimension
        """
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            adj=self.adj,
            num_gru_units=self._hidden_dim,
            output_dim=self._hidden_dim * 2,
            bias=1.0,
        )
        self.graph_conv2 = TGCNGraphConvolution(
            adj=self.adj,
            num_gru_units=self._hidden_dim,
            output_dim=self._hidden_dim,
            bias=0.0,
        )

    def forward(self, inputs, hidden_state):
        """Executes forward pass of TGCNCell

        Args:
            inputs ([type]): input features
            hidden_state ([type]): hidden state

        Returns:
            [type]: output prediction at that time step, new_hidden_state
        """
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))

        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)

        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))

        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        """Sets hyperparameter values for GRU module.

        Returns:
            [type]: input_dim, hidden_dim
        """
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


###############################################################################
class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, **kwargs):
        """Constructor of TGCN class for 1 TGCN module which take in adj matrix, 1 row time-step, pass them through TGCN cell then output the prediction.

        Args:
            adj ([type]): adjacency matrix
            hidden_dim (int): hidden dimension
        """
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)

    def forward(self, inputs):
        """Executes forward pass of the 1 TGCN Module 

        Args:
            inputs ([type]): input features

        Returns:
            [type]: output prediction.
        """
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        """Adds model specific arguments for TGCN module

        Args:
            parent_parser ([type]): parser

        Returns:
            [type]: parser
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        """Sets hyperparameter values for TGCN module

        Returns:
            [type]: input_dim, hidden_dim
        """
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
