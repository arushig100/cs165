import torch
import torch.nn as nn

class SequentialImageRNN2to2(nn.Module):
    """
    This class defines a recurrent neural network (RNN) model that takes a sequence of two image representations as input
    and predicts the next two image representations in the sequence.
    
    Attributes:
        rnn (nn.RNN): The RNN layer for sequential processing of input data.
        decoder (nn.Sequential): A feedforward neural network for decoding the hidden state of the RNN into the output space.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initializes the SequentialImageRNN2to2 model with the given parameters.
        
        Args:
            input_size (int): The size of each input image representation.
            hidden_size (int): The size of the RNN's hidden state.
            output_size (int): The size of each output image representation.
            num_layers (int, optional): The number of layers in the RNN. Defaults to 1.
        """
        super(SequentialImageRNN2to2, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """
        Defines the forward pass of the SequentialImageRNN2to2 model.
        
        Args:
            x (torch.Tensor): The input tensor containing a batch of sequences. Shape: [batch_size, 2 (seq_length), input_size].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two tensors representing the predicted next two image representations in the sequence.
        """
        output, hidden = self.rnn(x)  # Process the input sequence through the RNN.
        last_hidden_state = output[:, -1, :]  # Extract the last hidden state.
        
        predicted_representation1 = self.decoder(last_hidden_state)  # Decode the last hidden state to predict the next image representation.
        predicted_representation1_for_rnn = predicted_representation1.unsqueeze(1)  # Prepare the prediction for re-entry into the RNN.
        
        output, _ = self.rnn(predicted_representation1_for_rnn, hidden)  # Process the predicted representation through the RNN.
        last_hidden_state_after_prediction = output[:, -1, :]  # Extract the new last hidden state.
        
        predicted_representation2 = self.decoder(last_hidden_state_after_prediction)  # Decode the new last hidden state to predict another image representation.
        
        return predicted_representation1, predicted_representation2


class SequentialImageRNN2to1(nn.Module):
    """
    This class defines a recurrent neural network (RNN) model that takes a sequence of two image representations as input
    and predicts the next image representation in the sequence.
    
    Attributes:
        rnn (nn.RNN): The RNN layer for sequential processing of input data.
        decoder (nn.Sequential): A feedforward neural network for decoding the hidden state of the RNN into the output space.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initializes the SequentialImageRNN2to1 model with the given parameters.
        
        Args:
            input_size (int): The size of each input image representation.
            hidden_size (int): The size of the RNN's hidden state.
            output_size (int): The size of the output image representation.
            num_layers (int, optional): The number of layers in the RNN. Defaults to 1.
        """
        super(SequentialImageRNN2to1, self).__init__()  # Fixed incorrect class name in the original code snippet.
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """
        Defines the forward pass of the SequentialImageRNN2to1 model.
        
        Args:
            x (torch.Tensor): The input tensor containing a batch of sequences. Shape: [batch_size, seq_length, input_size].
        
        Returns:
            torch.Tensor: The predicted next image representation in the sequence.
        """
        output, _ = self.rnn(x)  # Process the input sequence through the RNN.
        final_hidden_state = output[:, -1, :]  # Extract the last hidden state.
        
        predicted_representation = self.decoder(final_hidden_state)  # Decode the last hidden state to predict the next image representation.
        
        return predicted_representation
