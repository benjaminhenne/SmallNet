import tensorflow as tf
import numpy as np
from settings import Settings
import smallnet_architecture as net
import sys
import os, time, re

def train(run, settings):
    print("########################")
    print("#     Build Network    #")
    print("########################")
    loader = settings.loader
    network = net.Smallnet(settings)

    print("########################")
    print("#       Training       #")
    print("########################")
    with tf.Session() as session:
        summary_writer = tf.summary.FileWriter(settings.summary_path + str(run), session.graph)
        saver = tf.train.Saver(max_to_keep=10000)

        # check if run already exits: if so continue run
        if os.path.isdir("./stored_weights/"+str(run)):
            print("[Info] Stored weights for run detected.")
            print("[Info] Loading weights...")
            saver.restore(session, tf.train.latest_checkpoint('./stored_weights/'+str(run)))
        else:
            session.run(tf.global_variables_initializer())

        # Initialize the global_step tensor
        tf.train.global_step(session, network.global_step)
        print(" Epoch | Val Acc | Avg Tr Acc | Avg. Loss | Avg. CrossEntropy | Avg. L1 Penalty | Time")
        print("-------+---------+------------+-----------+-------------------+-----------------+------------")
        for epoch in range(settings.epochs):
            t = time.time()

            ## Training
            losses = []
            penalties = []
            cross_entropies = []
            accuracies = []
            for train_inputs, train_labels in loader.get_training_batch(settings.batch_size):
                global_step = tf.train.global_step(session, network.global_step)
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if global_step % 50 == 0 else None
                run_metadata = tf.RunMetadata() if global_step % 50 == 0 else None
                _global_step, _xentropy, _penalty, _logits, _summaries, _, _loss, _accuracy = session.run(
                    [network.global_step, network.xentropy, network.penalty, network.logits, network.summaries,
                    network.update, network.loss, network.accuracy],
                    feed_dict={
                        network.inputs:train_inputs,
                        network.labels:train_labels,
                        network.learning_rate: settings.l_rate
                        },
                        options=run_options,
                        run_metadata=run_metadata)
                losses.append(_loss)
                penalties.append(_penalty)
                cross_entropies.append(_xentropy)
                accuracies.append(_accuracy)
                # write summaries
                if global_step % 50 == 0:
                    summary_writer.add_run_metadata(run_metadata, 'step%d' % global_step)
                summary_writer.add_summary(_summaries, global_step)

            # validation
            val_inputs, val_labels = next(loader.get_validation_batch(0))
            val_acc, _ = session.run([network.accuracy, network.loss],
                feed_dict={
                    network.inputs:val_inputs,
                    network.labels:val_labels,
                    network.learning_rate: 0.001})

            # Save model
            store_path = os.path.join("./stored_weights", str(run))
            os.makedirs(store_path, exist_ok=True)
            saver.save(session, os.path.join(store_path, "small_weights"), global_step=_global_step)

            #Printing Information
            t = time.time() - t
            minutes, seconds = divmod(t, 60)
            avg_loss = np.average(losses)
            avg_penalty = np.average(penalties)
            avg_cross_entropy = np.average(cross_entropies)
            avg_tr_acc = np.average(accuracies)
            #print(" Epoch | Val Acc | Avg TrAcc | Avg. CrossEntropy | Avg. L1 Penalty")
            print(" #{0:3d}  | {1:^7.3f} | {2:^10.3f} | {3:^9.3f} | {4:^17.3f} | {5:^15.3f} | {6:^3.0f}m {7:^4.2f}s".format(
                epoch + 1, val_acc, avg_tr_acc, avg_loss, avg_cross_entropy, avg_penalty, minutes, seconds))
            print("-------+---------+------------+-----------+-------------------+-----------------+------------")

def extract_number(f):
    s = re.findall(r"\d+$",f)
    return (int(s[0]) if s else -1,f)

def main(argv):
    settings = Settings()
    run = -1
    if len(argv) == 0:
        files = [d.name for d in os.scandir(settings.summary_path)]
        run = str(int(max(files,key=extract_number)) + 1)
    else:
        run = argv[0]
    if os.path.isdir(settings.summary_path + run):
        print('[Attention] The specified run already exists!')
        sys.exit()

    train(run, settings)

if __name__ == "__main__":
   main(sys.argv[1:])
