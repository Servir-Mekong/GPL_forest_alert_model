from __future__ import print_function
from __future__ import division 

# Silence futurewarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Silence warnings from the PyTorch Lightning evaluation metrics 
import logging
logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

import torch
import torch.onnx
import pandas as pd
import sklearn.metrics as sk_metrics
import torchvision.transforms as transforms
from torch.autograd import Variable
from barbar import Bar

# My Modules
from utils.model_dataset import AlertDataset
import utils.transforms as custom_transforms
import utils.loss_functions as losses
from utils.unet_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net

class model_trainer():
    
    def __init__(self, model, model_form_id, training_dir, dev_dir, test_dir, stats_dir, 
                 weights_dir, model_weights=None, image_size=256, batch_size=6, max_epochs=200, 
                 drop_lr_n_epochs = 5, print_error=True, log_error_every_n_epochs=5, 
                 device = "cuda:0", silent=False):

        self.input_network = model
        self.configured_network = None
        self.model_form_id = model_form_id
        self.training_dir = training_dir
        self.dev_dir = dev_dir
        self.test_dir = test_dir
        self.stats_dir = stats_dir
        self.weights_dir = weights_dir
        self.model_weights = model_weights
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.drop_lr_n_epochs = drop_lr_n_epochs
        self.print_error = print_error
        self.log_error_every_n_epochs = log_error_every_n_epochs
        self.device = device
        self.silent = silent
        
        # Initalize the lists used to store information about the network's accuracy
        self.training_lists = []
        self.training_loss = []
        self.dev_lists = []        
        return
    
    def __eval_net(self, dataloader, dataset_type, epoch):
        '''
        Calculate evaluation metrics using Pytorch Lightning classification metrics.
        '''
        # Set the network to evaluation mode
        self.configured_network.eval()
        
        # Define the evaluation metrics
        temp_iou = 0
        temp_accuracy = 0
        temp_f1_score = 0
        temp_precision = 0 
        temp_recall = 0
        
        # Stop the accumulation of gradients
        with torch.no_grad():

            # Iterate through the data in the dataloader
            for i, data in enumerate(Bar(dataloader)):
            
                # Extract the images and labels
                images, labels = data

                # Make sure the loss can be computed on the GPU
                images, labels = Variable(images).cuda(), Variable(labels).cuda()

                # Get the prediction
                outputs = self.configured_network(images)

                # Compute the the accuracy    
                predictions = torch.argmax(outputs, 1)
                
                # Reshape, move to the CPU, and convert to numpy
                predictions = predictions.reshape(-1).cpu().numpy()
                labels = labels.reshape(-1).cpu().numpy()

                # Compute the different loss scores
                temp_iou += sk_metrics.jaccard_score(labels, predictions, average='macro')
                temp_accuracy += sk_metrics.accuracy_score(labels, predictions)
                temp_f1_score += sk_metrics.f1_score(labels, predictions, average='macro')
                temp_precision += sk_metrics.precision_score(labels, predictions, average='macro')
                temp_recall += sk_metrics.recall_score(labels, predictions, average='macro')
        
        # Compute the average of the dataset across the dataset
        temp_iou = temp_iou / len(dataloader)
        temp_accuracy = temp_accuracy / len(dataloader)
        temp_f1_score = temp_f1_score / len(dataloader)
        temp_precision = temp_precision / len(dataloader)
        temp_recall = temp_recall / len(dataloader)

        # Reset the network to training mode
        self.configured_network.train()
        
        # Display the training information
        if self.print_error and not self.silent:
            print('IoU', temp_iou)
            print('Accuracy', temp_accuracy)
            print('F1 Score', temp_f1_score)
            print('Precision', temp_precision)
            print('Recall', temp_recall)
            
        # Log the information
        metrics_list = [epoch+1, temp_accuracy, temp_iou, temp_f1_score, temp_precision, temp_recall]
        if dataset_type == 'train':
            self.training_lists.append(metrics_list)
        elif dataset_type == 'dev':
            self.dev_lists.append(metrics_list)
        
        return None
    
    def __compute_dev_loss(self, dataloader):
        '''
        Compute a single loss value for the trainer to return
        '''
        # Set the network to evaluation mode
        self.configured_network.eval()
        
        # Define the evaluation metrics
        temp_iou = 0
        
        # Stop the accumulation of gradients
        with torch.no_grad():
            
            for i, data in enumerate(Bar(dataloader)):

                # Extract the images and labels
                images, labels = data

                # Make sure the loss can be computed on the GPU
                images, labels = Variable(images).cuda(), Variable(labels).cuda()

                # Get the prediction
                outputs = self.configured_network(images)
                
                # Compute the the accuracy    
                predictions = torch.argmax(outputs, 1)

                # Reshape, move to the CPU, and convert to numpy
                predictions = predictions.reshape(-1).cpu().numpy()
                labels = labels.reshape(-1).cpu().numpy()

                # Compute the different loss scores
                temp_iou += sk_metrics.jaccard_score(labels, predictions, average='macro')

        # Reset the network to training mode
        self.configured_network.train()
        
        return temp_iou / len(dataloader)
    
    def __format_and_export_logs(self, learning_rate):
        '''
        Write out the logs of error over the training and the test sets. 
        '''
        # Convert the model parameters to a dring
        learning_rate_str = str(learning_rate).replace('.', '_') 
        
        # Define the columns names
        column_names = ['Epoch', 'Accuracy', 'IoU', 'F1_score', 'Precision', 'Recall']
        
        # Format the two dataframes
        train_df = pd.DataFrame(self.training_lists, columns=column_names)
        train_loss_df = pd.DataFrame(self.training_loss, columns=['iter', 'loss', 'grad_l2'])
        dev_df = pd.DataFrame(self.dev_lists, columns=column_names)
        
        # Define the base output path
        path_base = learning_rate_str + '.csv'
        
        # Define the two output filenames
        training_path = self.stats_dir + '/' + self.model_form_id + '_' +'train__' + path_base
        training_loss_path = self.stats_dir + '/' + self.model_form_id + '_' + 'train_loss__' + path_base
        dev_path = self.stats_dir + '/' + self.model_form_id + '_' + 'dev__' + path_base
        
        # Write out the dataset
        train_df.to_csv(training_path, sep=',', index=False, line_terminator='\n')
        train_loss_df.to_csv(training_loss_path, sep=',', index=False, line_terminator='\n')
        dev_df.to_csv(dev_path, sep=',', index=False, line_terminator='\n')
        
        # Reset the log lists
        self.training_lists = []
        self.training_loss = []
        self.dev_lists = []
                
        return None

    def __save_model_weights(self, network, learning_rate):
        '''
        Save the model weights after the training is complete. 
        '''
        # Convert the model parameters to a dring
        learning_rate_str = str(learning_rate).replace('.', '_') 
        
        # Format the output name
        write_model_name = self.model_form_id + '__' + learning_rate_str + '.pth'
        
        # Input to the model
        x = torch.randn(1, 9, self.image_size, self.image_size, requires_grad=True).cuda()
        
        # Creat the ONNX model name
        onnx_model_name = self.model_form_id + '__' + learning_rate_str + '.onnx'
        
        # Create the input names
        inputs = ['inputs']
        
        # Outputs to the model
        output_names = ['outputs']
        
        # Export the model
        torch.onnx.export(network,   # model being run
                          x,                         # model input (or a tuple for multiple inputs)
                          onnx_model_name,           # where to save the model (can be a file or file-like object)
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=10,          # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names = inputs,   # the model's input names
                          output_names = output_names, # the model's output names
                          dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                        'output' : {0 : 'batch_size'}})
        
        # Save the model weights
        torch.save(network.state_dict(), self.weights_dir+'/'+write_model_name)
        
        return None

    def __get_data_loaders(self):
        '''
        Retrieve the two dataloaders. 
        '''
        # Define the transforms that need to be use
        data_transforms = transforms.Compose([
            custom_transforms.RandomRotateLeft2DTensor(p=0.5),
            custom_transforms.RandomRotateRight2DTensor(p=0.5),
            custom_transforms.RandomVerticalFlip2DTensor(p=0.5)
        ])
        
        # Load in the training datasets
        training_dataset = AlertDataset(self.training_dir, num_classes=2, transform=data_transforms)
        dev_dataset = AlertDataset(self.dev_dir, num_classes=2, transform=None)
        test_dataset = AlertDataset(self.test_dir, num_classes=2, transform=None)
        
        # Define the dataset loaders
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)
    
        return train_loader, dev_loader, test_loader
    
    def __compute_gradient_norm(self):
        '''
        Computes the L2 norm of the gradients
        '''
        total_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.configured_network.parameters())):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def train_model(self, learning_rate=0.01):
        '''
        Trains a model with the given hyperparameters.
        '''
        # Get the data loaders
        train_loader, dev_loader, test_loader = self.__get_data_loaders()
        
        # Instantiate the network -- note that every time the function is called
        # self.configured_network is overridden
        if not self.silent:
            print('Building model...')
        self.configured_network = self.input_network.to(torch.device(self.device))
        
        # Load the weights if given
        if self.model_weights is not None:
            self.configured_network.load_state_dict(torch.load(self.model_weights))
        
        # Set the network to training mode
        self.configured_network.train() 
        
        # Load the Optimizer
        optimizer = torch.optim.Adam(self.configured_network.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        
        # Load in the learning rate adjuster
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.drop_lr_n_epochs, gamma=0.1)
       
        # Create the summary writer
        criterion = losses.jaccard_loss
        
        if not self.silent:
            print('Starting training...')
            print('\nTraining for {epochs} epochs.'.format(epochs=self.max_epochs))
        
        # Initialize the iteration counter
        iteration_counter = 0
        
        # Loop through the training dataset self.max_epochs number of times
        for epoch in range(self.max_epochs):
            
            # Initialize the minibatch loss at zero
            loss = 0
            
            # Print out information about the current epoch
            if not self.silent:
                print('\nEpoch {} start:'.format(epoch + 1))
            
            # Ensure the gradients are set to zero before the epoch starts
            optimizer.zero_grad()
            
            for i, data in enumerate(Bar(train_loader)):

                # get the inputs
                inputs, labels = data
    
                # Wrap them in Variable
                # Shapes
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda().long()
                
                # Forward pass to get outputs
                outputs = self.configured_network(inputs)
                
                #print(outputs.shape, labels.shape)
                            
                # Compute the loss
                loss = criterion(outputs, labels)
                
                # Create the compute graph
                loss.backward()
                
                # Print out the loss ever N epochs -- for debugging
                if i % 100 == 0:
                    print('\nJaccard Score: ', losses.jaccard_score(outputs, labels).item(), "\nGradient: ", self.__compute_gradient_norm())
                
                # Log the loss for the current iteration
                iteration_counter += 1
                self.training_loss.append([iteration_counter, losses.jaccard_loss(outputs, labels).item(), self.__compute_gradient_norm()])
    
                # Optimize weights
                optimizer.step()
            
                # Reset the accumulated gradients and loss
                optimizer.zero_grad()

            # Step the learning (this happens in the epoch loop not the training loop)
            scheduler.step()
            
            # Print out training information
            if ((epoch+1) % self.log_error_every_n_epochs) == 0:
                if self.print_error and not self.silent:
                    print('\n--- Computing Training Error ---')
                self.__eval_net(train_loader, 'train', epoch)
                if self.print_error and not self.silent:
                    print('\n--- Computing Dev Error ---')
                self.__eval_net(dev_loader, 'dev', epoch)
                if self.print_error and not self.silent:
                    print('\n--- Computing Test Error ---')
                self.__eval_net(test_loader, 'Test', epoch)
                
        # Write out the error to a directory
        self.__format_and_export_logs(learning_rate)
        
        # Save the model
        self.__save_model_weights(self.configured_network, learning_rate)
                    
        # Compute the final loss value for the training fucntionb to return for an optimizer
        # Here, were using the IoU (Jaccard)
#        dev_loss = self.__compute_dev_loss(dev_loader)
        
        return #dev_loss 
    
if __name__ == "__main__":
    
    # Define the path to the training, testing, dev sets
    training_dir = "C://Users//John Kilbride//Desktop//temp_ssd_space//tensors//training_data_2"
    dev_dir = "C://Users//John Kilbride//Desktop//temp_ssd_space//tensors//val_data"
    test_dir = "C://Users//John Kilbride//Desktop//temp_ssd_space//tensors//test_data"
    
    # Define the directories two write out the S
    write_stats_dir = "C://Users//John Kilbride//Desktop//temp_ssd_space//model_run_tables"
    write_weights_dir = "C://Users//John Kilbride//Desktop//temp_ssd_space//model_weights"
    
    # Load in the model 
    model = AttU_Net(img_ch=9, output_ch=1)
    
    # Instantiate the model training class
    trainer = model_trainer(model = model, 
                            model_form_id = 'partial_dataset_', 
                            training_dir = training_dir, 
                            dev_dir =  dev_dir,
                            test_dir =  test_dir,
                            stats_dir = write_stats_dir,
                            weights_dir = write_weights_dir,
                            image_size = 64, 
                            batch_size = 128, 
                            max_epochs = 30,
                            drop_lr_n_epochs = 15,
                            print_error = True,
                            log_error_every_n_epochs = 10,
                            device = 'cuda:0'
                            )
    
    # Train the model   
    trainer.train_model(learning_rate=0.001)   
    print("Program complete")
    
    
    
    
    
    
    
    
    
    
    