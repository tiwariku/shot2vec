from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import model_fns as mf
import data_processing as dp

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)





class KerasWorker(Worker):
    def __init__(self, N_train=8192, N_valid=1024, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = 100
        self.corpus_filename = '../../assets/corpi/full_coords_bin_10'
        self.num_steps = 500
        self.gen_step_len = self.batch_size*self.num_steps

        (train,
         valid,
         test,
         self.vocabulary,
         self.play_to_id,
         self.id_to_play) = dp.corpus_to_keras(self.corpus_filename,
                                               pad_play=str({'Type':'Nothing'}))

        self.train_len = sum([len(game) for game in train])
        self.test_len = sum([len(game) for game in test])

        self.train_gen = mf.KerasBatchGenerator(train,
                                                self.num_steps,
                                                self.batch_size,
                                                self.vocabulary,
                                                skip_step=self.num_steps)

        self.test_gen = mf.KerasBatchGenerator(test,
                                               self.num_steps,
                                               self.batch_size,
                                               self.vocabulary,
                                               skip_step=self.num_steps)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        """
        hidden_size = config['hidden_size']
        dropout = config['dropout']
        learning_rate = config['learning_rate']

        model = Sequential()
        model.add(layers.Embedding(self.vocabulary,
                                   hidden_size,
                                   input_length=self.num_steps))
        model.add(layers.LSTM(hidden_size, return_sequences=True))
        model.add(layers.LSTM(hidden_size, return_sequences=True))
        model.add(layers.Dropout(dropout))
        model.add(layers.TimeDistributed(layers.Dense(self.vocabulary,
                                                      activation='softmax')))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'])

        history = model.fit_generator(self.train_gen.generate(),
                                      self.train_len//(self.gen_step_len),
                                      epochs=int(budget),
                                      verbose=0,
                                      validation_data=self.test_gen.generate(),
                                      validation_steps=self.gen_step_len)

        loss = model.evaluate_generator(self.train_gen.generate(),
                                        self.train_len//(self.gen_step_len))

        #import IPython; IPython.embed()
        return ({'loss': loss, # remember: HpBandSter always minimizes!
                 'info': {'': None,
                          'training history': history,
                         }})


    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('learning_rate',
                                            lower=1e-6,
                                            upper=1e-1,
                                            default_value='1e-2',
                                            log=True)

        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate',
                                                      lower=0.0,
                                                      upper=0.9,
                                                      default_value=0.5,
                                                      log=False)

        hidden_size = CSH.UniformIntegerHyperparameter('hidden_size',
                                                       lower=10,
                                                       upper=200,
                                                       default_value=32,
                                                       log=True)

        cs.add_hyperparameters([lr, dropout_rate, hidden_size])
        return cs




if __name__ == "__main__":
    worker = KerasWorker(run_id='0')
    cs = worker.get_configspace()
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)
    print(id2config[incumbent]['config'])
