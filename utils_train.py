import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import time
import models


model_inputs_dict = {
    'MLP': ['du_dy'],
    'MLP-Prior': ['du_dy'],
    'MLP-BC': ['du_dy', 'y+'],
    'MLP-Re': ['du_dy', 'Re_tau'],
    'MLP-NL': ['y+', 'index', 'DU_DY', 'Y'],
    'MLP-Re-Prior': ['du_dy', 'Re_tau'],
    'MLP-BC-Re': ['du_dy', 'Re_tau', 'y+'],
    'MLP-BC-Re-Prior': ['du_dy', 'Re_tau', 'y+'],
    'MLP-BC-Re-NL': ['y+', 'index', 'DU_DY', 'Y', 'Re_tau']
}


class ChannelDataset(Dataset):

    def __init__(self, df, input_labels, target_label):

        toArray = lambda x: np.atleast_1d(x)  # convert scalar to array
        toTensor = lambda x: torch.from_numpy(x)  # convert array to tensor
        toFloat = lambda x: x.float() if x.dtype == torch.float64 else x  # convert double to float

        self.df = df.applymap(toArray).applymap(toTensor).applymap(toFloat)

        self.inputs = self.df[input_labels].values
        self.target = self.df[target_label].values

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        inputs = list(self.inputs[index])
        target = list(self.target[index])[0]

        return inputs, target


class ModelTrainer:
    """Model training """

    def __init__(self):
        self.model = None
        self.loss_history = None
        self.input_labels = None
        self.target_label = None
        self.df_train = None
        self.df_val = None
        self._criterion = None
        self._optimizer = None
        self._training_log = []

    def set_target_variable(self, target_label):
        self.target_label = target_label

    def set_model(self, model_name, **kwargs):
        self.model_name = model_name
        self.model_kwargs = kwargs
        model = {
            'MLP': models.MLP,
            'MLP-Prior': models.MLP_Prior,
            'MLP-BC': models.MLP_BC,
            'MLP-Re': models.MLP_Re,
            'MLP-Re-Prior': models.MLP_Re_Prior,
            'MLP-NL': models.MLP_NL,
            'MLP-BC-Re': models.MLP_BC_Re,
            'MLP-BC-Re-Prior': models.MLP_BC_Re_Prior,
            'MLP-BC-Re-NL': models.MLP_BC_Re_NL
        }[self.model_name]
        self.model = model(**kwargs)
        self.input_labels = model_inputs_dict[self.model_name]

    def set_criterion(self, criterion):
        self._criterion = criterion

    def set_optimizer(self, optimizer_fn, **kwargs):
        self._optimizer_fn = optimizer_fn
        optim_fn = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'Adadelta': torch.optim.Adadelta,
            'Adagrad': torch.optim.Adagrad,
            'SparseAdam': torch.optim.SparseAdam,
            'Adamax': torch.optim.Adamax,
            'ASGD': torch.optim.ASGD,
            'LBFGS': torch.optim.LBFGS,
            'RMSprop': torch.optim.RMSprop,
            'Rprop': torch.optim.Rprop}[self._optimizer_fn]
        self._optimizer = optim_fn(self.model.parameters(), **kwargs)
        
    def set_config(self, batch_size=10, print_freq=500, max_epochs=2000, min_epochs=0, earlystopping=True, patience=30):
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.earlystopping = earlystopping
        self.patience = patience

    def print_summary(self):
        print('\n### MODEL ### \n\n', self.model)
        print('\n### OPTIMIZER ### \n\n', self._optimizer)
        print('\n### CRITERION ### \n\n', self._criterion)
        print('\n### TRAINING ### \n')
        for l in self._training_log:
            print(l)

    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_name': self.model_name,
            'model_kwargs': self.model_kwargs,
            'state_dict': self.model.state_dict(),
            'optimizer_fn': self._optimizer_fn,
            'optimizer_state_dict': self._optimizer.state_dict(),
            'criterion': self._criterion,
            'loss_history': self.loss_history,
            'df_train': self.df_train,
            'df_val': self.df_val,
            'input_labels': self.input_labels,
            'target_label': self.target_label,
            'training_log': self._training_log
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.set_model(checkpoint.get('model_name'), **checkpoint.get('model_kwargs'))
        self.model.load_state_dict(checkpoint.get('state_dict'))
        self.set_optimizer(checkpoint.get('optimizer_fn'), lr=1.0)
        self._optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))
        self.set_criterion(checkpoint.get('criterion'))
        self.loss_history = checkpoint.get('loss_history')
        self.df_train = checkpoint.get('df_train')
        self.df_val = checkpoint.get('df_val')
        self.input_labels = checkpoint.get('input_labels')
        self.target_label = checkpoint.get('target_label')
        self._training_log = checkpoint.get('training_log', [])

    def predict(self, df):

        dataloader = DataLoader(ChannelDataset(df, self.input_labels, self.target_label),
                                batch_size=1, shuffle=False)
        preds = []
        targets = []

        self.model.eval()
        with torch.no_grad():
            # preds = torch.cat([self.model(*inputs) for inputs, target in dataloader])
            for inputs, target in dataloader:
                preds.append(self.model(*inputs))
                targets.append(target)

        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()

        return preds, targets

    def _step(self, data_batch, optimizer=None):

        inputs_batch, target_batch = data_batch
        pred_batch = self.model(*inputs_batch)
        loss = self._criterion(pred_batch, target_batch)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item(), len(target_batch)

    def fit(self, df_train, df_val, df_test):
            #batch_size=1,
            #print_freq=1, max_epochs=1000, min_epochs=0, earlystopping=False, patience=0):

        # Record start time
        start_time = time.time()

        self.df_train = df_train
        self.df_val = df_val

        # Convert dataframes to datasets
        ds_train = ChannelDataset(df_train, self.input_labels, self.target_label)
        ds_val = ChannelDataset(df_val, self.input_labels, self.target_label)
        ds_test = ChannelDataset(df_test, self.input_labels, self.target_label)

        # Convert datasets to dataloaders
        trainloader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True)
        valloader = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False)
        testloader = DataLoader(ds_test, batch_size=self.batch_size, shuffle=False)

        # Initialize loss_history
        if self.loss_history is None:
            self.loss_history = {'train': [], 'val': [], 'test': []}

        # Initialize trackers
        start_epoch = len(self.loss_history['train'])
        epoch = start_epoch
        keep_going = True
        best_loss = 1e15
        wait = 0
        stopped_epoch = 0

        # Print title of diagnostic statistics
        print('\n{0: >6} {1: >10} {2: >15} {3: >15} {4: >15}'.format(
            'epoch', 'time(s)', 'train_loss', 'val_loss', 'test_loss'))

        while keep_going:  # In each epoch

            # Record start time of current epoch
            epoch_start_time = time.time()

            # Set model to train mode
            self.model.train()

            # Loop over trainloader
            losses, nums = zip(
                *[self._step(data_batch, self._optimizer) for data_batch in trainloader]
            )
            train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # Set model to eval mode
            self.model.eval()

            # Loop over valloader
            with torch.no_grad():
                losses, nums = zip(*[self._step(data_batch) for data_batch in valloader])
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # Loop over testloader
            with torch.no_grad():
                losses, nums = zip(*[self._step(data_batch) for data_batch in testloader])
            test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # Append losses
            self.loss_history['train'].append(train_loss)
            self.loss_history['val'].append(val_loss)
            self.loss_history['test'].append(test_loss)

            # Check convergence
            if epoch < self.min_epochs:
                pass
            elif epoch >= self.max_epochs:
                keep_going = False
            else:
                if self.earlystopping:
                    current_loss = val_loss

                    if (current_loss - best_loss) < -1e-15:
                        best_loss = current_loss
                        wait = 0
                    else:
                        if wait >= self.patience:
                            keep_going = False
                            stopped_epoch = epoch
                        wait += 1
                else:
                    pass

            # Record time spent on current epoch
            epoch_time = time.time() - epoch_start_time

            # Print diagnostic statistics
            if epoch % self.print_freq == 0 or keep_going == False:
                print('{0:6d} {1:10.2f} {2:15.5f} {3:15.5f} {4:15.5f}'.format(
                    epoch, epoch_time, train_loss, val_loss, test_loss))

            epoch += 1

        # Print end epoch, total training time
        if stopped_epoch > 0:
            print('\nTerminated training for early stopping at Epoch {0:d}'.format(stopped_epoch))
        print('\nTotal training time: {0:.2f} sec'.format(time.time()-start_time))

        # Append info to trainig log
        log = 'Epoch {}-{}: batch_size={}, earlystopping={}, patience={}'.format(
        start_epoch, epoch-1, self.batch_size, self.earlystopping, self.patience)
        self._training_log.append(log)
        
        return self.loss_history # new


def plot_loss_history(loss_history, loglog=False):

    plt.plot(loss_history['train'], 'k-')
    plt.plot(loss_history['val'], 'r--')
    plt.plot(loss_history['test'], 'b:')
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation', 'test'])

    #print('Train loss: {0:.1e}'.format(loss_history['train'][-1]))
    #print('Val loss: {0:.1e}'.format(loss_history['val'][-1]))
    #print('Test loss: {0:.1e}'.format(loss_history['test'][-1]))


def calc_r2(y_true, y_predicted):
    """
    Calculate R^2 score
    :param y_true: true value
    :param y_predicted: predicted value
    """
    assert y_true.shape == y_predicted.shape, "Shape mismatch"
    SS_res = np.sum(np.square(y_true - y_predicted))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true, axis=0)))
    r2 = 1 - SS_res / SS_tot
    return r2
