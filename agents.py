#########################
# Purpose: Mimics a benign agent in the federated learning setting and sets up the master agent 
########################
import os
import numpy as np
# tf.set_random_seed(777)
# np.random.seed(777)
import keras
import tensorflow.python.keras.backend as K
from utils.mnist import model_mnist
from utils.census_utils import census_model_1

from utils.eval_utils import eval_minimal
from multiprocessing import Process, Manager
import global_vars as gv
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')
# tf.disable_v2_behavior()
# tf.get_logger().setLevel('ERROR')
import time
# from utils.dpOptimizer import DPAdamOptimizer, DPGradientDescentOptimizer, DPAdagradGaussianOptimizer, DPAdamGaussianOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.dp_query import gaussian_query, quantile_adaptive_clip_sum_query

# from utils.dist_utils import collate_weights

# from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
# from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

# def compute_epsilon(steps):
#     """Computes epsilon value for given hyperparameters."""
#     orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
#     sampling_probability = gv.args.B / 60000
#     rdp = compute_rdp(
#         q=sampling_probability,
#         noise_multiplier=FLAGS.noise_multiplier,
#         steps=steps,
#         orders=orders)
#     # Delta is set to 1e-5 because MNIST has 60000 training points.
#     return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

# import tensorflow_privacy

# from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gv.mem_frac)

def alarm(i, X_test, Y_test, t, gpu_id, return_dict, shared_weights=None, previous_local_weights=None):
    args = gv.args
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print("alarm process_%s start" % i)
    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    previous_local_weights = np.load(gv.dir_name + 'ben_weights_%s_t%s.npy' % (i, t - 1), allow_pickle=True)
    local_success, local_loss = eval_minimal(X_test, Y_test, previous_local_weights)
    global_success, global_loss = eval_minimal(X_test, Y_test, shared_weights)
    if local_success > (global_success + local_success * args.client_c):
        print("alarm!_%s" % i)
        return_dict["alarm" + str(i)] = 1
    else:
        print("no alarm_%s" % i)
        return_dict["alarm" + str(i)] = 0

def agent(i, X_shard, Y_shard, t, gpu_id, return_dict, X_test, Y_test, lr=None):
    K.set_learning_phase(1)
    gv.init()
    args = gv.args
    if lr is None:
        lr = args.eta
    # print('Agent %s on GPU %s' % (i,gpu_id))
    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # new added block---------------------------------------
    if 'siren' in args.gar:
        X_test_original = X_test
        Y_test_original = Y_test
        X_test = X_shard[0:int(len(X_shard) / 10)]
        X_shard = X_shard[int(len(X_shard) / 10) + 1:len(X_shard) - 1]
        Y_test = Y_shard[0:int(len(Y_shard) / 10)]
        Y_shard = Y_shard[int(len(Y_shard) / 10) + 1:len(Y_shard) - 1]
        Y_test = np.argmax(Y_test, axis=1)
    # ------------------------------------------------------

    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    shard_size = len(X_shard)
    weights = 0
    if t == 0:
        print("t = %s" % t)
        weights = shared_weights
        print('loaded shared weights')


    else:
        if 'siren' in args.gar:
            print("t = %s" % t)
            p = Process(target=alarm, args=(i, X_test, Y_test, t, gpu_id, return_dict))
            p.start()
            p.join()
            if return_dict["alarm" + str(i)] == 1:
                previous_local_weights = np.load(gv.dir_name + 'ben_weights_%s_t%s.npy' % (i, t - 1), allow_pickle=True)
                weights = previous_local_weights
                print("loaded previous weights")
            else:
                weights = shared_weights
        else:
            weights = shared_weights

    if 'siren' in args.gar:
        with open('output/alarm_%s.txt' % i, 'a') as f:
            f.write('%s\n' % return_dict["alarm" + str(i)])
        X_test = X_test_original
        Y_test = Y_test_original
    # if i == 0:
    #     # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    #     eval_success, eval_loss = eval_minimal(X_test,Y_test,shared_weights)
    #     print('Global success at time {}: {}, loss {}'.format(t,eval_success,eval_loss))

    if args.steps is not None:
        num_steps = args.steps
    else:
        num_steps = int(args.E) * shard_size / args.B
    num_steps = int(num_steps)
    # with tf.device('/gpu:'+str(gpu_id)):
    if args.dataset == 'census':
        x = tf.placeholder(shape=(None,
                              gv.DATA_DIM), dtype=tf.float32)
        # y = tf.placeholder(dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)
    else:
        x = tf.placeholder(shape=(None,
                                  gv.IMAGE_ROWS,
                                  gv.IMAGE_COLS,
                                  gv.NUM_CHANNELS), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)

    if 'MNIST' in args.dataset:
        agent_model = model_mnist(type=args.model_num)
    elif args.dataset == 'census':
        agent_model = census_model_1()
    elif args.dataset == 'CIFAR-10':
        agent_model = model_mnist(type=args.model_num)

    logits = agent_model(x)

    if args.dataset == 'census':
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=y, logits=logits)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))
    else:
        # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            # labels=y, logits=logits))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=y, logits=logits)
        if args.dp:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    prediction = tf.nn.softmax(logits)

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(loss)
        if args.dp:
            noise_multiplier = args.dp_mul
            # optimizer = DPAdamGaussianOptimizer(learning_rate=lr).minimize(loss)
            # optimizer = dp_optimizer.DPAdamGaussianOptimizer(
            #     l2_norm_clip=10.0,
            #     noise_multiplier=noise_multiplier,
            #     learning_rate=lr
            # ).minimize(loss, global_step=tf.train.get_global_step())
            query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipSumQuery(
                initial_l2_norm_clip=10.0,
                noise_multiplier=noise_multiplier,
                target_unclipped_quantile=0.5,
                learning_rate=args.eta,
                clipped_count_stddev=0.5,
                expected_num_records=args.B,
                geometric_update=True)
            optimizer = dp_optimizer.DPAdamOptimizer(
                dp_sum_query=query
            ).minimize(loss, global_step=tf.train.get_global_step())
            
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=lr).minimize(loss)
        if args.dp:
            noise_multiplier = args.dp_mul
            # optimizer = DPGradientDescentOptimizer(learning_rate=lr).minimize(loss)
            # optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
            #     l2_norm_clip=10.0,
            #     noise_multiplier=noise_multiplier,
            #     learning_rate=lr
            # ).minimize(loss, global_step=tf.train.get_global_step())
            query = quantile_adaptive_clip_sum_query.QuantileAdaptiveClipSumQuery(
                initial_l2_norm_clip=10.0,
                noise_multiplier=noise_multiplier,
                target_unclipped_quantile=0.5,
                learning_rate=args.eta,
                clipped_count_stddev=0.5,
                expected_num_records=args.B,
                geometric_update=True)
            optimizer = dp_optimizer.DPGradientDescentOptimizer(
                dp_sum_query=query
            ).minimize(loss, global_step=tf.train.get_global_step())

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        config.intra_op_parallelism_threads = 12
        config.inter_op_parallelism_threads = 2
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    elif args.k == 1:
        sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
#-------------------------------------------
    agent_model.set_weights(weights)
    start_offset = 0
    if args.steps is not None:
        start_offset = (t * args.B * args.steps) % (shard_size - args.B)

    for step in range(num_steps):
        offset = (start_offset + step * args.B) % (shard_size - args.B)
        X_batch = X_shard[offset: (offset + args.B)]
        Y_batch = Y_shard[offset: (offset + args.B)]
        Y_batch_uncat = np.argmax(Y_batch, axis=1)
        _, loss_val = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch_uncat})
        if step % 1000 == 0:
            print('Agent %s, Step %s, Loss %s, offset %s' % (i, step, np.mean(loss_val), offset))
            # local_weights = agent_model.get_weights()
            # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
            # print('Agent {}, Step {}: success {}, loss {}'.format(i,step,eval_success,eval_loss))

    local_weights = agent_model.get_weights()
    local_delta = local_weights - shared_weights
    # _, _, flatten_delta = collate_weights(local_delta)
    # print("delta norm:", np.linalg.norm(flatten_delta))

    if args.dp:
        eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            len(X_shard), args.B, noise_multiplier, args.E*(t+1), 1e-5)
        print('For delta=1e-5, the current epsilon is: %.2f' % eps)

    # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    eval_success, eval_loss = eval_minimal(X_test, Y_test, local_weights)

    print('Agent {}: success {}, loss {}'.format(i, eval_success, eval_loss))
    with open('output/client_%s.txt' % i, 'a') as f:
        f.write('Agent {}: success {}, loss {}\n'.format(i,eval_success,eval_loss))

    return_dict[str(i)] = np.array(local_delta)

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (i, t), local_delta)
    if t > 0 and os.path.exists(gv.dir_name + 'ben_delta_%s_t%s.npy' % (i,t-1)):
        os.remove(gv.dir_name + 'ben_delta_%s_t%s.npy' % (i,t-1))
    np.save(gv.dir_name + 'ben_weights_%s_t%s.npy' % (i, t), local_weights)
    if t > 0 and os.path.exists(gv.dir_name + 'ben_weights_%s_t%s.npy' % (i,t-1)):
        os.remove(gv.dir_name + 'ben_weights_%s_t%s.npy' % (i,t-1))

    sess.close()
    return


def master():
    K.set_learning_phase(1)

    args = gv.args
    print('Initializing master model')
    config = tf.ConfigProto(gpu_options=gv.gpu_options)
    config.intra_op_parallelism_threads = 12
    config.inter_op_parallelism_threads = 2
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    if 'MNIST' in args.dataset:
        global_model = model_mnist(type=args.model_num)
    elif args.dataset == 'census':
        global_model = census_model_1()
    elif args.dataset == 'CIFAR-10':
        global_model = model_mnist(type=args.model_num)
    # if args.model_num < 9:
    #     global_model.summary()
    global_model.summary()
    global_weights_np = global_model.get_weights()
    np.save(gv.dir_name + 'global_weights_t0.npy', global_weights_np)

    return
