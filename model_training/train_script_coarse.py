#!/usr/local/bin/python3
'''
Submission script to train a coarse grained version of shot2vec. Getting
started on beluga
@tiwariku
2019-07-23
'''
import data_processing as dp
import model_fns as mf

if __name__ == '__main__':
    NUM_STEPS = 400
    BATCH_SIZE = 10
    HIDDEN_SIZE = 20
    NUM_EPOCHS = 1

    STRIP_FN = dp.strip_name_only

    CORPUS_FILENAME = 'coarse_corpus'
    MODEL_FILENAME = '2019-07-23_coarse'
    CORPUS = dp.get_corpus(2010, 2010)
    dp.pickle_it('coarse_corpus', obj=CORPUS)

    (TRAIN_DATA,
     VALID_DATA,
     TEST_DATA,
     VOCABULARY,
     PLAY_TO_ID,
     ID_TO_PLAY) = dp.corpus_to_keras(CORPUS_FILENAME,
                                      pad_play=str({'Type':'Nothing'}))
    TRAIN_LEN = sum([len(game) for game in TRAIN_DATA])
    TEST_LEN = sum([len(game) for game in TEST_DATA])

    TRAIN_GENERATOR = mf.KerasBatchGenerator(TRAIN_DATA,
                                             NUM_STEPS,
                                             BATCH_SIZE,
                                             VOCABULARY,
                                             skip_step=NUM_STEPS)

    TEST_GENERATOR = mf.KerasBatchGenerator(TEST_DATA,
                                            NUM_STEPS,
                                            BATCH_SIZE,
                                            VOCABULARY,
                                            skip_step=NUM_STEPS)

    MODEL = mf.make_LSTM_RNN(VOCABULARY, HIDDEN_SIZE, NUM_STEPS)
    MODEL.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    CP_FILEPATH = MODEL_FILENAME+'-{epoch:02d}.hdf5'
    CHECKPOINTER = mf.keras.callbacks.ModelCheckpoint(filepath=CP_FILEPATH,
                                                      verbose=1)
    MODEL.fit_generator(TRAIN_GENERATOR.generate(),
                        TRAIN_LEN//(BATCH_SIZE*NUM_STEPS),
                        NUM_EPOCHS,
                        validation_data=TEST_GENERATOR.generate(),
                        validation_steps=TEST_LEN//(BATCH_SIZE*NUM_STEPS),
                        callbacks=[CHECKPOINTER],
                        initial_epoch=0)
