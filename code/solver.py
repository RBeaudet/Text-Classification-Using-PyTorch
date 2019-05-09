import time

import torch
from torch import functional as F


class Solver():
    """
    Ecapsulates all the logic necessary for training text classification
    models.
    
    The Solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.
    
    - To train a model, construct Solver instance, pass the model, dataset, and 
    various options (learning rate, batch size, etc) to the
    constructor. 
    - Call the train() method to train the model.
    - Instance variable solver.loss_history contains a list of all losses 
    encountered during training and the instance variables 
    - Instance variables solver.train_acc_history and solver.val_acc_history are lists of the
    accuracies of the model on the training and validation set at each epoch.
    
    Example usage :
    model = Model(*args)
    solver = Solver(model, 
                    loader_train,
                    loader_val,
                    optimizer)
    solver.train()
    """
    
    def __init__(self, model,  optimizer, loader_train, loader_val, **kwargs):
        """
        Required arguments:
        - model: A model constructed from PyTorch nn.Module.
        - optimizer: An Optimizer object we will use to train the model.
        - loader_train: An Iterator object on which iterating to construct batches of 
          training data.
        - loader_val: An Iterator object on which iterating to construct batches of 
          validation data.
          
        Optional arguments:
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - save_model: Boolean; if set to True then save best model.
        """
        self.model = model
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_val = loader_val

        # Unpack arguments
        self.verbose = kwargs.pop('verbose', False)
        self.save_model = kwargs.pop('save_model', False)
        
        self._reset()
        self._select_device()
        
        
    def _reset(self):
        """
        Reset some variables for book-keeping:
        - best validation accuracy
        - loss history
        - train accuracy history
        - validation accuracy history
        - best model parameters
        """
        self.best_val_accuracy = 0
        self.loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.best_params = {}


    def _select_device(self, verbose=True):
        """
        Select device e.g. CPU / GPU
        """
        use_gpu = False
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if verbose:
            print('Using device:', device)

        self.device = device
        
    
    def train(self, print_every=10, epochs=1):
        """
        Train a model using the PyTorch Module API.

        Arguments:
        - print_every: (Optional) Print training accuracy every print_every iterations.
        - epochs: (Optional) A Python integer giving the number of epochs to train for.

        Returns: Nothing, but prints model accuracies during training.
        """

        # Move the model parameters to CPU / GPU
        model = self.model.to(device=self.device)
        optimizer = self.optimizer
        
        # Initialize iteration
        t = 0

        for epoch in range(epochs):
            start = time.time()
            for train_batch in self.loader_train:

                # Put model to training mode
                model.train()

                # Load x and y
                x = train_batch.text.transpose(1, 0)  # reshape to [batch_size, len_seq]
                y = train_batch.target.type(torch.LongTensor)

                # Move to device, e.g. CPU
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # Compute scores and softmax loss
                scores = model(x)
                loss = F.cross_entropy(scores, y)

                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                optimizer.zero_grad()

                # Backwards pass: compute the gradient of the loss with
                # respect to each parameter of the model.
                loss.backward()

                # Update the parameters of the model using the gradients
                # computed by the backwards pass.
                optimizer.step()
                
                # Save loss
                self.loss_history.append(loss.item())

                # Display information
                if self.verbose and t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, self.loss_history[-1]))
                    acc = self.compute_accuracy(validation=True)
                    print('Accuracy :', acc)
                    print()
                
                t += 1
                
            end = time.time()
            print('Epoch {0} / {1}, time = {2} secs'.format(epoch, epochs, end-start))
            
            # Compute train and val accuracy at the end of each epoch.
            train_accuracy = self.compute_accuracy(validation=False)
            val_accuracy = self.compute_accuracy(validation=True)
            
            self.train_accuracy_history.append(train_accuracy)
            self.val_accuracy_history.append(val_accuracy)
            
            # Print useful information
            if self.verbose:
                print('(Epoch %d / %d) Train acc: %f; Val acc: %f' % (epoch, epochs, 
                                                                      train_accuracy, val_accuracy))

            # Keep track of the best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                # update best params
                self.best_params['state_dict'] = model.state_dict().copy()
                self.best_params['optimizer'] = optimizer.state_dict().copy()
                    
        # Save best model
        if self.save_model:
            self._save_model('/Users/robin/Projects/zelros/', 
                             self.best_params['state_dict'], 
                             self.best_params['optimizer'])
                
                
    def compute_accuracy(self, validation=True):
        """
        Compute accuracy of a model.
        
        Arguments:
        - validation: (Optional) If True, compute accuracy on the validation dataset.
        """
        if validation:
            loader = self.loader_val
        else:
            loader = self.loader_train
                        
        num_correct = 0
        num_samples = 0

        # Set model to evaluation mode : This has any effect only on certain modules. 
        # For example, behaviors of dropout layers during train or test differ.
        self.model.eval()

        # Tell PyTorch not to build computational graphs
        with torch.no_grad():
            for batch in loader:

                # Load x and y
                x = batch.text.transpose(1, 0)  # reshape to [batch_size, len_seq]
                y = batch.target.type(torch.LongTensor)

                # Move to device, e.g. CPU
                x = x.to(device=self.device)  
                y = y.to(device=self.device)

                # Compute scores and predictions
                scores = self.model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        return acc
    
    
    def _save_model(self, model_path, model_dict, optimizer_dict):
        """
        Save model parameters if we have to interrupt the training.
        Parameters are saved in a dictionary.
        
        Required arguments:
        - model_path: where to save the model.
        - model_dict: Python dictionary that maps each layer to its parameter tensor.
        - optimizer_dict: Python dictionary that contains info about the optimizer's 
          states and hyperparameters used.
        """
    
        state = {
            'state_dict': model_dict,   # model.state_dict()
            'optimizer' : optimizer_dict   # optimizer.state_dict()
        }
        filename = model_path + 'best_model.pkl'
        torch.save(state, filename)
        print('Model saved to %s' % filename)
