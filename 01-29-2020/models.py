import torch
import torch.nn as nn


class NetworkStructure:
    """A class to define the layer structure for the neural network"""

    def __init__(self):
        self.num_layers = 1  # Number of hidden layers
        self.num_nodes = 10  # Number of nodes per hidden layer
        self.input_len = 1  # Number of nodes in input layer
        self.output_len = 1  # Number of nodes in output layer
        self.activation = nn.ReLU()  # Activation function of hidden layers

    def set_num_layers(self, num_layers):
        self.num_layers = num_layers
        return self

    def set_num_nodes(self, num_nodes):
        self.num_nodes = num_nodes
        return self

    def set_input_len(self, input_len):
        self.input_len = input_len
        return self

    def set_output_len(self, output_len):
        self.output_len = output_len
        return self

    def set_activation(self, activation):
        self.activation = activation
        return self

    def get_layer_sizes(self):
        sizes = [self.input_len] + [self.num_nodes] * self.num_layers + [self.output_len]
        in_sizes, out_sizes = sizes[:-1], sizes[1:]
        return in_sizes, out_sizes


class MLP(nn.Module):
    '''Multilayer Perceptron (MLP)'''

    def __init__(self, structure=None):
        '''
        :param structure: network structure
        '''
        super(MLP, self).__init__()

        # Set network structure
        if structure is None:
            structure = NetworkStructure()
        self.structure = structure

        # Obtain layer sizes from network structure
        in_sizes, out_sizes = self.structure.get_layer_sizes()

        # Construct linear layers
        self.linears = nn.ModuleList()
        for n_in, n_out in zip(in_sizes, out_sizes):
            self.linears.append(nn.Linear(n_in, n_out))

        # Set activation function
        self.activation = self.structure.activation

    def forward(self, x):
        '''
        :param x: input to network  [batch_size, input_len]
        '''

        for l in self.linears[:-1]:
            x = self.activation(l(x))   # x: [batch_size, num_nodes]
        x = self.linears[-1](x)         # x: [batch_size, output_len]

        return x
    
    
    
class Prior(nn.Module):
    '''stand alone piror network'''

    def __init__(self, prior_structure=None, scale=1):
        '''
        :param structure: network structure
        '''
        super(Prior, self).__init__()
        #self.net = nn.Sequential(MLP(structure), nn.Sigmoid())  # ensure output in between [0,1]

        # Set networks structure
        if prior_structure is None:
            self.prior_structure = NetworkStructure().set_num_layers(5).set_num_nodes(50).set_output_len(1).set_activation(nn.ELU())

        # Layer sizes from network structure
        prior_in_sizes, prior_out_sizes = self.prior_structure.get_layer_sizes() # prior
        
        # Construct linear layers
        self.linears = nn.ModuleList()
        for p_n_in, p_n_out in zip(prior_in_sizes, prior_out_sizes):
            self.linears.append(nn.Linear(p_n_in, p_n_out))
        
        # Set activation function
        self.prior_activation = self.prior_structure.activation

    def forward(self, x):

        for l in self.linears[:-1]:
            x = self.prior_activation(l(x))   # x: [batch_size, num_nodes]
        x = self.linears[-1](x)         # x: [batch_size, output_len]

        return x

    
    
class G(nn.Module):
    '''Function with the property that G(0)=0'''

    def __init__(self, beta=0.1):
        super(G, self).__init__()
        self.beta = beta

    def forward(self, x):
        return 1. - torch.exp(-self.beta * x)

    def extra_repr(self):
        return 'beta={}'.format(self.beta)


    
class AlphaNet(nn.Module):
    '''Network that predicts alpha from y+'''

    def __init__(self, structure=None, alpha0=1):
        '''
        :param structure: network structure
        :param alpha0: boundary condition for alpha(0)
        '''
        super(AlphaNet, self).__init__()

        self.net = nn.Sequential(MLP(structure), nn.Sigmoid())  # ensure output in between [0,1]

        assert alpha0 == 0 or alpha0 == 1
        self.alpha0 = alpha0
        self.g = G()

    def forward(self, x):
        '''
        :param x: input to network  [batch_size, 1]
        '''
        alpha = {0: self.net(x) * self.g(x),
                 1: 1 - self.net(x) * self.g(x)}[self.alpha0]
        return alpha        # alpha: [batch_size, 1]
    
    
    
class MLP_Prior(nn.Module):
    '''MLP with a parallel prior network'''
    
    def __init__(self, structure=None, prior_structure=None, scale=1):
        super(MLP_Prior, self).__init__()

        self.prior_net = Prior(prior_structure, scale)
        self.net = MLP(structure)
        self.scale = scale

    def forward(self, x):

        prior_output = self.prior_net(x)                   
        net_output = self.net(x)
        output = torch.add(net_output, self.scale, prior_output)

        return output

    

class MLP_Re(nn.Module):
    '''MLP with Reynolds number injection '''

    def __init__(self, structure=None, Re_input_layers=(0,)):
        '''
        :param structure: network structure
        :param Re_input_layers: at which layers to insert the Re input
        '''
        super(MLP_Re, self).__init__()

        # Set network structure
        if structure is None:
            structure = NetworkStructure()
        self.structure = structure

        # Obtain layer sizes from network structure
        in_sizes, out_sizes = self.structure.get_layer_sizes()

        # Adjust layer sizes based on specified Re input layers
        assert max(Re_input_layers) < len(in_sizes) and min(Re_input_layers) >= 0
        self.Re_input_layers = Re_input_layers
        for i in self.Re_input_layers:
            in_sizes[i] += 1

        # Construct linear layers
        self.linears = nn.ModuleList()
        for n_in, n_out in zip(in_sizes, out_sizes):
            self.linears.append(nn.Linear(n_in, n_out))

        # Set activation function
        self.activation = self.structure.activation

    def forward(self, x, Re):
        '''
        :param x: input to network  [batch_size, input_len]
        :param Re: Re input         [batch_size, 1]
        '''

        if 0 in self.Re_input_layers:
            x = torch.cat((x, Re), dim=1)       # x: [batch_size, input_len+1]
        for i, l in enumerate(self.linears[:-1]):
            x = self.activation(l(x))           # x: [batch_size, num_nodes]
            if i+1 in self.Re_input_layers:
                x = torch.cat((x, Re), dim=1)   # x: [batch_size, num_nodes+1]
        x = self.linears[-1](x)                 # x: [batch_size, output_len]

        return x
    
    

class MLP_BC(nn.Module):
    '''MLP with boundary condition enforcement'''

    def __init__(self, structure=None):
        '''
        :param structure: network structure
        '''
        super(MLP_BC, self).__init__()

        self.net = MLP(structure)
        self.g = G()

    def forward(self, x, y_plus):
        '''
        :param x: input to network                  [batch_size, input_len]
        :param y_plus: y+ input for enforcing BC    [batch_size, 1]
        '''
        return self.net(x) * self.g(y_plus)

    
class MLP_NL(nn.Module):
    '''MLP with non-locality'''

    def __init__(self, structure=None, alphanet_structure=None, alpha0=1):
        '''
        :param structure: network structure
        :param alphanet_structure: structure of alphanet
        :param alpha0: boundary condition for alpha(0)
        '''
        super(MLP_NL, self).__init__()

        self.alphanet = AlphaNet(alphanet_structure, alpha0)
        self.net = MLP(structure)

    def forward(self, y_plus, i, DU_DY, Y):
        '''
        :param y_plus: input to alphanet    [batch_size, 1]
        :param i:                           [batch_size, 1]
        :param DU_DY:                       [batch_size, 1, N]
        :param Y:                           [batch_size, 1, N]
        '''

        alpha = self.alphanet(y_plus)                   # alpha: [batch_size, 1]
        D_alpha_u = calc_D_alpha_u(i, alpha, DU_DY, Y)  # D_alpha_u: [batch_size, 1]
        output = self.net(D_alpha_u)                    # output: [batch_size, net.output_len]

        return output
    

    
class MLP_Re_Prior(nn.Module):
    '''MLP-Re with a parallel prior network'''
    
    def __init__(self, structure=None, Re_input_layers=(0,), prior_structure=None, scale=1):
        super(MLP_Re_Prior, self).__init__()
        
        self.scale = scale
        self.prior_net = Prior(prior_structure, scale)
        self.net = MLP_Re(structure, Re_input_layers)

    def forward(self, x, Re):

        prior_output = self.prior_net(x)                   
        net_output = self.net(x, Re)
        output = torch.add(net_output, self.scale, prior_output)

        return output


    
class MLP_BC_Re(nn.Module):
    '''MLP with boundary condition enforcement and Reynolds number injection'''

    def __init__(self, structure=None, Re_input_layers=(0,)):
        '''
        :param structure: network structure
        :param Re_input_layers: at which layers to insert the Re input
        '''
        super(MLP_BC_Re, self).__init__()

        self.net = MLP_Re(structure, Re_input_layers)
        self.g = G()

    def forward(self, x, Re, y_plus):
        '''
        :param x: input to network                  [batch_size, input_len]
        :param Re: Re input                         [batch_size, 1]
        :param y_plus: y+ input for enforcing BC    [batch_size, 1]
        '''
        return self.net(x, Re) * self.g(y_plus)



class MLP_BC_Re_Prior(nn.Module):
    
    def __init__(self, structure=None, Re_input_layers=(0,), prior_structure=None, scale=1):
        super(MLP_BC_Re_Prior, self).__init__()
        
        self.scale = scale
        self.net = MLP_BC_Re(structure, Re_input_layers)
        self.prior_net = Prior(prior_structure, scale)

    def forward(self, x, Re, y_plus):
        
        net_output = self.net(x, Re, y_plus)
        prior_output = self.prior_net(x)
        output = torch.add(net_output, self.scale, prior_output)
    
        return output

    
class MLP_BC_Re_NL(nn.Module):
    '''MLP with boundary condition enforcement, Reynolds number injection, and non-locality'''

    def __init__(self, structure=None, Re_input_layers=(0,), alphanet_structure=None, alpha0=1):
        '''
        :param structure: network structure
        :param Re_input_layers: at which layers to insert the Re input
        :param alphanet_structure: structure of alphanet
        :param alpha0: boundary condition for alpha(0)
        '''
        super(MLP_BC_Re_NL, self).__init__()

        self.alphanet = AlphaNet(alphanet_structure, alpha0)
        self.net = MLP_BC_Re(structure, Re_input_layers)

    def forward(self, y_plus, i, DU_DY, Y, Re):
        '''
        :param y_plus: input to alphanet [batch_size, 1]
        :param i: [batch_size, 1]
        :param DU_DY: [batch_size, 1, N]
        :param Y: [batch_size, 1, N]
        :param Re: Re input [batch_size, 1]
        '''

        alpha = self.alphanet(y_plus)                   # alpha: [batch_size, 1]
        D_alpha_u = calc_D_alpha_u(i, alpha, DU_DY, Y)  # D_alpha_u: [batch_size, 1]
        output = self.net(D_alpha_u, Re, y_plus)        # output: [batch_size, net.output_len]

        return output


class Swish(nn.Module):
    """Swish activation function"""

    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

    def extra_repr(self):
        return 'beta={}'.format(self.beta)


    
def calc_D_alpha_u(n_batch, alpha_n_batch, DU_DY_batch, Y_batch):

    return torch.stack(
        [D_alpha_u_n(n.squeeze(), alpha_n, DU_DY, Y) for n, alpha_n, DU_DY, Y in
        zip(n_batch, alpha_n_batch, DU_DY_batch, Y_batch)]
    )


def D_alpha_u_n(n, alpha_n, DU_DY, Y):
    '''
    Compute D^alpha(y)_y u(y) at y=y_n.
    (Note: the fractional derivative at y_0 is not defined when 0<alpha(y_0)<1.
    In such cases, the output is 0.)

    :param n      : integer n in [0, N].
    :param alpha_n: alpha(y_n). Must be in [0, 1].
    :param DU_DY  : du/dy(y_n) for n = 0, 1, ..., N.
    :param Y      : y_n for n = 0, 1, ..., N.
    :return       : scalar.
    '''
    torch_gamma = lambda x: torch.exp(torch.mvlgamma(x, p=1))

    def _D_0_u_n(n, DU_DY, Y):

        # integrate du/dy from y_0 to y_n using the composite trapezoidal rule
        res = torch.dot(Y[1:n+1]-Y[:n], DU_DY[1:n+1]+DU_DY[:n]) / 2.
        return res

    def _D_1_u_n(n, DU_DY, Y):
        return DU_DY[n]

    def _D_alpha_u_n(n, alpha_n, DU_DY, Y):
        a = alpha_n
        fac = 1./torch_gamma(2-a)

        b = lambda k: (Y[n]-Y[k])**(1-a) - (Y[n]-Y[k+1])**(1-a)

        kk = torch.arange(n-1)
        sum_ = torch.dot(b(kk), DU_DY[kk]) + (Y[n]-Y[n-1])**(1-a) * DU_DY[n-1]

        return fac * sum_

    if n == 0:
        if alpha_n == 0:
            res = _D_0_u_n(n, DU_DY, Y).view(alpha_n.shape)
        elif alpha_n == 1:
            res = _D_1_u_n(n, DU_DY, Y).view(alpha_n.shape)
        else:
            res = torch.full(alpha_n.shape, 0)

    else:
        if alpha_n == 0:
            res = _D_0_u_n(n, DU_DY, Y).view(alpha_n.shape)
        elif alpha_n == 1:
            res = _D_1_u_n(n, DU_DY, Y).view(alpha_n.shape)
        else:
            res = _D_alpha_u_n(n, alpha_n, DU_DY, Y)

    return res
