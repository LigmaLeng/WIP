from synapse import cp, pd,\
                    Eve,\
                    Dense,\
                    ReLU, Softmax, Sigmoid, cross_entropy, cse_softmax_prime
from synapse.data import *
import gc
"""
Raw
len:  (13901,) min_words: 1  max_words: 767
Raw_valid
len:  (1737,) min_words: 1  max_words: 903
Raw_test
len:  (1738,) min_words: 2  max_words: 584
NaN in rawfile idx: 7418  JOB-2019-0002789
"""
TF_LEN = 500
EMBED_LEN = 384
OUTPUT_SIZE = 10

tf_network = [
        Dense(TF_LEN, 256, "first"),
        ReLU(),
        Dense(256, 256, "fc2"),
        Sigmoid(),
        Dense(256, OUTPUT_SIZE, "fc3"),
        Softmax()
    ]

embed_network = [
        Dense(EMBED_LEN, 200, "first"),
        ReLU(),
        Dense(200, 200, "fc2"),
        Sigmoid(),
        Dense(200, OUTPUT_SIZE, "fc3"),
        Softmax()
    ]

def run():
    # run_validation(minibatch=8)
    tfidf_model = Eve(tf_network, cross_entropy, cse_softmax_prime, alpha=1* 1e-3)
    emb_model = Eve(embed_network, cross_entropy, cse_softmax_prime, alpha=1* 1e-3)
    run_learning_curve(filename="mini_batches", model1=tfidf_model, model2=emb_model,epochs=5)

def run_validation(minibatch:int,filename:str=None, model1=None, model2=None, epochs=1):
    y_train, y_test = load_labels()
    x_train, z_train = load_train()

    model1.train_ensemble(twin=model2, x_train=x_train, z_train=z_train,\
                    y_train=y_train, minibatch=minibatch, epochs=epochs)

    x_test, z_test = load_valid()
    predictions, labels = model1.test_ensemble(twin=model2, x_test=x_test,\
                                                z_test=z_test, y_test=y_test)
    cp.savez(file=f"ens_validations/{filename}", predictions=predictions,labels=labels)


def run_learning_curve(filename:str="def", model1=None, model2=None,epochs=1):
    y_train, y_valid = load_labels()
    x_train, z_train = load_train()
    x_valid, z_valid= load_valid()


    mini_schedule = [1,4,8,16,32,64,128,256,512,1024]
    lr_schedule = [x for x in range(-1,9,1)]
    patience_schedule = [20,18,16,10,8,4,2,2,1,1]

    for i in range(len(mini_schedule)):
        if mini_schedule[i] not in [256,512,1024]:continue
        model1.update_alpha(1e-4*(5/3)**lr_schedule[i])
        model2.update_alpha(1e-4*(5/3)**lr_schedule[i])
        t_loss, t_predicted_val, t_labels = model1.train_ensemble(twin=model2, x_train=x_train, y_train=y_train, z_train=z_train,\
                                                                  minibatch=mini_schedule[i], epochs=epochs,patience=patience_schedule[i])
        
        preds, labels,loss = model1.test_ensemble(twin=model2, x_test=x_valid,\
                                                z_test=z_valid, y_test=y_valid)

        model1.zero_grad()
        model2.zero_grad()
        cp.savez(file=f"ens_curves/{filename}/{mini_schedule[i]}", batch_preds=t_predicted_val,batch_labels=t_labels,batch_loss=t_loss,\
                                                    batchv_preds=preds,batchv_labels=labels,batchv_loss=loss)


def one_hot(y, num_labels=OUTPUT_SIZE):
    one_hot_label = cp.zeros((num_labels, y.shape[0]), dtype=cp.uint16)
    for i, label in enumerate(y):
        one_hot_label[label, i] = 1
    return one_hot_label

def load_labels():
    labels = cp.load(Raw.labels)
    valid_labels = cp.asarray(labels["valid_labels"], dtype=cp.uint16)
    train_labels = cp.asarray(labels["train_labels"], dtype=cp.uint16)
    y_train = one_hot(train_labels)
    y_test = one_hot(valid_labels)
    return y_train, y_test

def load_train():
    x_train = cp.load(Tfidf.train)
    z_train = cp.load(Embeddings.train)
    return x_train, z_train

def load_valid():
    x_valid = cp.load(Tfidf.valid)
    z_valid = cp.load(Embeddings.valid)
    return x_valid, z_valid

if __name__ == "__main__":
    run()




