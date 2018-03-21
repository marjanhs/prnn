
# author Marjan Hosseinia
import numpy as np
import os, argparse
from sklearn.model_selection import StratifiedKFold
import util, model
os.environ['KERAS_BACKEND'] = 'theano'


def train_and_evaluate_model(model_, data_train, y_train, data_test, y_test, epochs, inter_model, batch, verbose=1):
    '''
    trains and evaluates the model
    :param model_: NN model
    :param data_train: training data
    :param y_train: labels of training set
    :param data_test:  test data
    :param y_test: labels of test set
    :param epochs:
    :param inter_model: itemnediate model to get the output of the fusion layer for plotting
    :param batch:
    :param verbose: keras verbose paramets of fit function
    :return:
    '''
    [l_x_train, r_x_train] = data_train
    hist = model_.fit([l_x_train, r_x_train], y_train,
                epochs=epochs, batch_size=batch, verbose=verbose)
    #print(hist)
    [l_x_test, r_x_test] = data_test
    loss_train, acc_train = model_.evaluate([l_x_train, r_x_train], y_train, batch_size=batch)
    loss_test, acc_test = model_.evaluate([l_x_test, r_x_test], y_test, batch_size=batch)

    intermediate_output_train = inter_model.predict([l_x_train, r_x_train])
    intermediate_output_test = inter_model.predict([l_x_test, r_x_test])

    print('train error :  {0} train accuracy:  {1} '.format(loss_train, acc_train))
    print('test error :  {0} test accuracy:  {1} '.format(loss_test, acc_test))
    return [loss_train,acc_train], [loss_test, acc_test],[intermediate_output_train,intermediate_output_test]


def get_args():
    '''
    get arguments from command line
    :return: a dic of all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', action='store', default='2015.cnn', help='dataset name')
    parser.add_argument('-bs', action='store', default=1, help='batch size', type=int)
    parser.add_argument('-do', action='store', default=0.2, help='dropout')
    parser.add_argument('-hs', action='store', default=8, help='hidden layer size', type=int)
    parser.add_argument('-nf', action='store', default=5, help='# of folds in CV', type=int)
    parser.add_argument('-ep', action='store', default=50, help='# of epochs', type=int)
    parser.add_argument('-ms', action='store', default=1000, help='maximum sequence length', type=int)
    parser.add_argument('-vb', action='store', default=1, help='verbose: 0, 1 or 2', type=int)

    #print parser.print_help()
    results = parser.parse_args()
    print(results)
    return vars(results)


def CV(args):
    '''
    k-fold Cross-Validation
    :param args: model arguments
    '''

    # loading model  parameters
    MAX_SEQUENCE_LENGTH = args['ms']
    embeddings_index = util.load_embedding('glove.6B.100d.txt')
    EMBEDDING_DIM = 100
    drops = args['do']
    batch = args['bs']
    hidden = args['hs']
    n_folds = args['nf']
    epochNo = args['ep']
    ds_id = args['ds']
    verbose = args['vb']

    # loading data

    data, labels, word_idx, id_all = util.load_data(dataset_id=ds_id,isonefile=False,
                                                     MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=None)

    # saving bins info
    i = 0
    f_bin = open(ds_id+'.interbins', 'w')
    f_bin.close()
    f_train = open(ds_id+'.interout_train', 'wt')
    f_train.close()
    f_test = open(ds_id+'.interout_test', 'wt')
    f_test.close()
    id_all = np.array(id_all)

    # Cross-Validation
    avg_acc_train, avg_acc_test, avg_error_train, avg_error_test =0,0,0,0
    for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
        f_bin = open(ds_id+'.interbins', 'a')
        np.savetxt(f_bin, [id_all[train_index]], fmt='%s')
        np.savetxt(f_bin, [id_all[test_index]], fmt ='%s')
        f_bin.close()
        print("size of train index ", len(train_index))
        print("size of test index ", len(test_index))
        print ("Running Fold %d/%d " % (i+1, n_folds))
        my_model = None  # Clearing the NN.
        my_model ,inter_model = model.model(drop=drops, hidden_units=hidden, word_index=word_idx,
                                            embedding_index=embeddings_index, EMBEDDING_DIM=EMBEDDING_DIM,
                                            MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
        [data_l, data_r] = data
        my_data_train = [data_l[train_index], data_r[train_index]]
        my_data_test = [data_l[test_index], data_r[test_index]]
        labels = np.asarray(labels)
        [loss_train, acc_train], [loss_test, acc_test],[inter_out_train,inter_out_test] = train_and_evaluate_model\
            (my_model, my_data_train,  labels[train_index], my_data_test,  labels[test_index], epochNo, inter_model,
             batch, verbose)

        a_train = labels[train_index].reshape(labels[train_index].shape[0],-1)
        a_test = labels[test_index].reshape(labels[test_index].shape[0],-1)
        print (inter_out_train.shape, a_train.shape)
        inter_out_train = np.concatenate((inter_out_train,a_train),axis=1)
        inter_out_test = np.concatenate((inter_out_test,a_test),axis=1)

        # updating bins info
        f_train = open(ds_id+'.interout_train', 'at')
        np.savetxt(f_train, inter_out_train)
        f_train.close()
        f_test = open(ds_id+'.interout_test', 'at')
        np.savetxt(f_test, inter_out_test)
        f_test.close()

        #  results
        avg_acc_train += acc_train
        avg_acc_test += acc_test
        avg_error_train += loss_train
        avg_error_test += loss_test
        i += 1
    print ("avg acc train , test :", avg_acc_train/n_folds, avg_acc_test/n_folds)
    print ("avg error train , test :", avg_error_train/n_folds, avg_error_test/n_folds)


if __name__=="__main__":
    params = get_args()
    CV(params)


