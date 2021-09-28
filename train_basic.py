from RL_train import iterator, init_hyperparams
import pickle

init_hyperparams['contrast'] = 0.0
init_hyperparams['sat'] = 0.0
init_hyperparams['bright'] = 0.0
init_hyperparams['cutlength'] = 0.0
init_hyperparams['cutholes'] = 0.0
init_hyperparams['momentum'] = 0.9
init_hyperparams['percent_valid'] = 0.2
init_hyperparams['warmup_epochs'] = 5
init_hyperparams['patience'] = 60
init_hyperparams['max_epoch'] = 100

for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]:
    for batch_size in [64, 128, 256, 512, 1024, 2048, 4096]:
        init_hyperparams['learning_rate'] = lr
        init_hyperparams['batch_size'] = batch_size
        save_filename = "%s_%s" % (lr, batch_size)
        with open('%s.ser' % save_filename, 'wb') as fp:
            pickle.dump(dict(), fp)
        DNN = iterator(1, 0, save_filename, 0.0)
        while(True):
            try:
                next(DNN)
            except StopIteration:
                break
        print("test obtained by using lr = %s, batch_size = %s" % (lr, batch_size))