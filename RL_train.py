import sys
import time
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.models import resnet18

from corrupted_loaders import create_loaders

# Should make training go faster for large models
cudnn.benchmark = True  
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)






# RL - Hyper-parameters default values for the warming train epochs
init_hyperparams = {}
# Tuned hyper-parameters
init_hyperparams['inscale'] = 0.05
init_hyperparams['hue'] = 0.05
init_hyperparams['contrast'] = 0.05
init_hyperparams['sat'] = 0.05
init_hyperparams['bright'] = 0.05
init_hyperparams['cutlength'] = 4.
init_hyperparams['cutholes'] = 1.
init_hyperparams['learning_rate'] = 0.05
# Fixed hyper-parameters
init_hyperparams['momentum'] = 0.9
init_hyperparams['percent_valid'] = 0.2
init_hyperparams['batch_size'] = 128
init_hyperparams['warmup_epochs'] = 5
init_hyperparams['patience'] = 60
init_hyperparams['max_epoch'] = 100





# RL - Encapsulation of Resnet train code using an iterator function
def iterator(train_steps, valid_steps, save_filename, corruption):
    global cnn, cnn_optimizer, hyperparams
    # RL - Hyper-parameters default values for the warming train epochs 
    hyperparams = init_hyperparams.copy()

    # Enables computation on GPU
    device = torch.device("cuda")

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)

    ###############################################################################
    # Data Loading/Processing
    ###############################################################################
    train_loader, valid_loader, test_loader = create_loaders(hyperparams, corruption, hyper=True)
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)

    ###############################################################################
    # Saving
    ###############################################################################
    def model_save(fn):
        with open(fn, 'wb') as f:
            torch.save([cnn, cnn_optimizer], f)

    def model_load(fn):
        global cnn, cnn_optimizer
        with open(fn, 'rb') as f:
            cnn, cnn_optimizer = torch.load(f)

    ###############################################################################
    # Model/Optimizer
    ###############################################################################
    cnn = resnet18(num_classes = 10)
    cnn = cnn.to(device)
    if torch.cuda.device_count() > 1:
        print("%s GPUs: use of DataParallel" % torch.cuda.device_count())
        cnn = nn.DataParallel(cnn)
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=hyperparams['learning_rate'], momentum=hyperparams['momentum'])


    ###############################################################################
    # Evaluation
    ###############################################################################
    def evaluate(loader):
        global cnn, cnn_optimizer, hyperparams
        cnn.eval()    # Change model to 'eval' mode.
        correct = total = loss = 0.
        loader.dataset.reset_hparams()
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                pred = cnn(images)
                probabilities = torch.nn.functional.softmax(pred, dim=0)
                loss += F.cross_entropy(probabilities, labels, reduction='sum').item()
                _, hard_pred = torch.max(probabilities, 1)
                total += labels.size(0)
                correct += (hard_pred == labels).sum().item()

        accuracy = correct / total
        mean_loss = loss / total
        return mean_loss, accuracy


    ###############################################################################
    # Optimization step
    ###############################################################################
    def next_batch(data_iter, data_loader, curr_epoch):
        """Load next minibatch."""
        try:
            images, labels = data_iter.next()
        except StopIteration:
            curr_epoch += 1
            data_iter = iter(data_loader)
            images, labels = data_iter.next()

        images, labels = images.to(device), labels.to(device)
        return images, labels, data_iter, curr_epoch


    def optimization_step(data_iter, data_loader, curr_epoch, hyper=False):
        # RL - Use of global variables to keep modifications done in this function
        global cnn, cnn_optimizer, hyperparams
        cnn_optimizer.zero_grad(set_to_none=True)
        cnn_optimizer.param_groups[0]['lr'] = hyperparams['learning_rate']
        data_loader.dataset.set_hparams(hyperparams)

        images, labels, data_iter, curr_epoch = next_batch(data_iter, data_loader, curr_epoch)

        # Apply input transformations.
        inscale = hyperparams['inscale']
        noise = torch.rand(images.size(0), device=device)
        scaled_noise = ((1 + inscale) - (1 / (1 + inscale))) * noise + (1/(1 + inscale))
        images = images * scaled_noise[:,None,None,None]

        pred = cnn(images)
        probabilities = torch.nn.functional.softmax(pred, dim=0)
        loss = F.cross_entropy(probabilities, labels)
        loss.backward()

        if not hyper:
            cnn_optimizer.step()

        # Calculate number of correct predictions.
        _, hard_pred = torch.max(probabilities, 1)
        num_correct = (hard_pred == labels).sum().item()
        num_points = labels.size(0)
        step_stats = {'loss': loss.item(), 'acc': num_correct/num_points}

        return data_iter, curr_epoch, step_stats







    ###############################################################################
    # Training Loop
    ###############################################################################
    # Bookkeeping stuff.
    train_step = valid_step = global_step = wup_step = 0
    curr_train_epoch = train_epoch = valid_epoch = 0
    start_time = time.time()
    # RL - Initialisation of the dictionnary of metrics and hyper-parameters values
    stats_to_yield = {  'learning_rate' : [], 'inscale' : [], 'hue' : [], 'contrast' : [], 'sat' : [], 'bright' : [], 'cutlength' : [], 'cutholes' : [], \
                        'global_step' : [], 'val_acc' : [], 'val_loss' : [], 'acc' : [], 'loss' : [], \
                        'current_epoch' : 0}

    # Warmup for specified number of epochs. Do not tune hyperparameters during this time.
    cnn.train()
    while train_epoch < hyperparams['warmup_epochs']:
        train_iter, train_epoch, stats = optimization_step(train_iter, train_loader, train_epoch)

        wup_step += 1
        global_step += 1

        if curr_train_epoch != train_epoch:
            val_loss, val_acc = evaluate(valid_loader)

            print('=' * 80)
            print('Train Epoch: {} | Val Loss: {:.3f} | Val acc: {:.3f}'.format(train_epoch, val_loss, val_acc))
            print('=' * 80)

            curr_train_epoch = train_epoch



    # Actual training of the classifier
    best_val_loss = []
    stored_loss = float('inf')
    patience_elapsed = 0
    try:
        # Enter main training loop. Alternate between optimizing on training set for
        # train_steps and on validation set for valid_steps.
        while patience_elapsed < hyperparams['patience'] and curr_train_epoch < hyperparams['max_epoch']:
            # Check whether we should use training or validation set.
            cycle_pos = (train_step + valid_step) % (train_steps + valid_steps)
            hyper = cycle_pos >= train_steps

            # Do a step on the training set.
            if not hyper:
                cnn.train()
                curr_train_epoch = train_epoch
                train_iter, train_epoch, stats = (optimization_step(train_iter, train_loader,train_epoch))
                changed_epoch = (curr_train_epoch != train_epoch)

                if global_step > 0:
                    print('Global Step: {} Train Epoch: {} \tTrain step:{} \tLoss: {:.3f} Accuracy: {:.3f} lr: {:.4e}'.format(global_step, train_epoch, train_step, stats['loss'], stats['acc'], hyperparams['learning_rate']))
                    # RL - Saving of the train metrics and hyper-parameters values that will be presented to the agent
                    stats_to_yield['global_step'].append(global_step)
                    stats_to_yield['acc'].append(stats['acc'])
                    stats_to_yield['loss'].append(stats['loss'])

                train_step += 1

            # Do a step on the validation set.
            else:
                cnn.eval()
                valid_iter, valid_epoch, stats = (optimization_step(valid_iter, valid_loader, valid_epoch, hyper=True))

                if global_step > 0:
                    print('Global Step: {} Valid Epoch: {} \t Valid Step {} \tLoss: {:.6f} Accuracy: {:.3f}'.format(global_step, valid_epoch, valid_step, stats['loss'], stats['acc']))
                    # RL - Saving of the validation metrics and hyper-parameters values that will be presented to the agent
                    stats_to_yield['global_step'].append(global_step)
                    stats_to_yield['val_acc'].append(stats['acc'])
                    stats_to_yield['val_loss'].append(stats['loss'])
                    stats_to_yield['current_epoch'] = train_epoch
                    
                    # RL - Presentation to the agent of the metrics and hyper-parameters values
                    for key in stats_to_yield:
                        if key in hyperparams:
                            stats_to_yield[key].append(hyperparams[key])
                    yield stats_to_yield

                    # RL - Loading of the new hyper-parameters values given by the agent
                    with open('%s.ser' % save_filename, 'rb') as fp:
                        new_hparam = pickle.load(fp)
                    for key in new_hparam:
                        hyperparams[key] = new_hparam[key]

                    # RL - Reset of the dictionnary of metrics and hyper-parameters values
                    stats_to_yield = {  'learning_rate' : [], 'inscale' : [], 'hue' : [], 'contrast' : [], 'sat' : [], 'bright' : [], 'cutlength' : [], 'cutholes' : [], \
                                        'global_step' : [], 'val_acc' : [], 'val_loss' : [], 'acc' : [], 'loss' : [], \
                                        'current_epoch' : 0}
                
                valid_step += 1

            global_step += 1

            # If just completed an epoch on the training set, check the validation loss.
            if changed_epoch:
                changed_epoch = False  # Reset changed_epoch back to False
                val_loss, val_acc = evaluate(valid_loader)

                print('=' * 80)
                print('Val Loss: {:.3f} | Val acc: {:.3f}'.format(val_loss, val_acc))
                print('=' * 80)

                best_val_loss.append(val_loss)

                if val_loss < stored_loss:
                    model_save('best_checkpoint.pt')
                    print('Saving model (new best validation)')
                    sys.stdout.flush()
                    stored_loss = val_loss
                    patience_elapsed = 0
                else:
                    patience_elapsed += 1


    except KeyboardInterrupt:
        print('=' * 89)
        print('Exiting from training early')
        sys.stdout.flush()


    # Load the best saved model.
    model_load('best_checkpoint.pt')

    # Run on val and test data.
    val_loss, val_acc = evaluate(valid_loader)
    test_loss, test_acc = evaluate(test_loader)

    print('=' * 89)
    print('| End of training | val loss {:8.5f} | val acc {:8.5f} | test loss {:8.5f} | test acc {:8.5f}'.format(val_loss, val_acc, test_loss, test_acc))
    print('=' * 89)
    sys.stdout.flush()