from skmultiflow.evaluation import EvaluatePrequential
from timeit import default_timer as timer
import time

import warnings
from skmultiflow.visualization.fgt_evaluation_visualizer import FGTEvaluationVisualizer

from numpy import unique
from skmultiflow.utils import constants

class FGTEvaluatePrequential(EvaluatePrequential):
    
    def __init__(self,
                 n_wait=200,
                 fgt_freq = 1000,
                 max_samples=100000,
                 batch_size=1,
                 pretrain_size=200,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 data_points_for_classification=False):
        # Evaluator configuration
        self.n_wait = 0
        self.max_samples = 0
        self.batch_size = 0
        self.pretrain_size = 0
        self.max_time = 0
        self.metrics = []
        self.output_file = None
        self.show_plot = False
        self.restart_stream = True
        self.test_size = 0
        self.dynamic_test_set = False
        self.data_points_for_classification = False

        # Metrics
        self.mean_eval_measurements = None
        self.current_eval_measurements = None
        self._data_dict = None
        self._data_buffer = None
        self._file_buffer = ''
        self._file_buffer_size = 0

        # Misc
        self._method = None
        self._task_type = None
        self._output_type = None
        self._valid_configuration = False
        self.model_names = None
        self.model = None
        self.n_models = 0
        self.stream = None
        self._start_time = -1
        self._end_time = -1

        self.visualizer = None
        self.n_sliding = 0
        self.global_sample_count = 0
        
        self._method = 'prequential'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        self.data_points_for_classification = data_points_for_classification

        if not self.data_points_for_classification:
            if metrics is None:
                self.metrics = [constants.ACCURACY, constants.KAPPA]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                else:
                    raise ValueError("Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        else:
            if metrics is None:
                self.metrics = [constants.DATA_POINTS]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                    self.metrics.append(constants.DATA_POINTS)
                else:
                    raise ValueError("Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        self.restart_stream = restart_stream
        self.n_sliding = n_wait
        self.fgt_freq = fgt_freq
        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")
        
         
         
         
        
    
    def _train_and_test(self):
        """ Method to control the prequential evaluation.
        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.
        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In
        the future, when BaseRegressor is created, it could be an extension from that
        class as well.
        """
        
        self._start_time = timer()
        self._end_time = timer()
        print('Prequential Evaluation')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        first_run = True
        if self.pretrain_size > 0:
            print('Pre-training on {} sample(s).'.format(self.pretrain_size))

            X, y = self.stream.next_sample(self.pretrain_size)

            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION:
                    # Training time computation
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values)
                    self.running_time_measurements[i].compute_training_time_end()
                elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=unique(self.stream.target_values))
                    self.running_time_measurements[i].compute_training_time_end()
                else:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y)
                    self.running_time_measurements[i].compute_training_time_end()
                self.running_time_measurements[i].update_time_measurements(self.pretrain_size)
            self.global_sample_count += self.pretrain_size
            first_run = False

        update_count = 0
        print('Evaluating...')
        samples_fgt = self.pretrain_size
        while ((self.global_sample_count < actual_max_samples) & (self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            
            
            try:
                X, y = self.stream.next_sample(self.batch_size)
                
                
                if X is not None and y is not None:
                    # Test
                    prediction = [[] for _ in range(self.n_models)]
                    for i in range(self.n_models):
                        
                        
                                
                                
                        
                        try:
                            
                            # Testing time
                            self.running_time_measurements[i].compute_testing_time_begin()
                            prediction[i].extend(self.model[i].predict(X))
                            self.running_time_measurements[i].compute_testing_time_end()
                        except TypeError:
                            raise TypeError("Unexpected prediction value from {}"
                                            .format(type(self.model[i]).__name__))
                    self.global_sample_count += self.batch_size

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.current_eval_measurements[j].add_result(y[i], prediction[j][i])
                    self._check_progress(actual_max_samples)

                    # Train
                    if first_run:
                        for i in range(self.n_models):
                            if self._task_type != constants.REGRESSION and \
                               self._task_type != constants.MULTI_TARGET_REGRESSION:
                                # Accounts for the moment of training beginning
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y, self.stream.target_values)
                                # Accounts the ending of training
                                self.running_time_measurements[i].compute_training_time_end()
                            else:
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y)
                                self.running_time_measurements[i].compute_training_time_end()

                            # Update total running time
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            self.running_time_measurements[i].compute_training_time_begin()
                            self.model[i].partial_fit(X, y)
                            self.running_time_measurements[i].compute_training_time_end()
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                            
                    samples_fgt += 1
                    
                    if samples_fgt % self.fgt_freq == 0:
                        for i in range(self.n_models):
                            if self.model[i].fgt == True:
                                    
                                    self.model[i].forget_last_randomized()
                                    
                               

                    if ((self.global_sample_count % self.n_wait) == 0 or
                            (self.global_sample_count >= self.max_samples) or
                            (self.global_sample_count / self.n_wait > update_count + 1)):
                        
                        if prediction is not None:
                            self._update_metrics()
                        update_count += 1

                self._end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break
        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print('Done')

        if self.restart_stream:
            self.stream.restart()

        return self.model
    
    def evaluate(self, stream, model, image_name, model_names=None):
        """ Evaluates a model or set of models on samples from a stream.

        Parameters
        ----------
        stream: Stream
            The stream from which to draw the samples.

        model: skmultiflow.core.BaseStreamModel or sklearn.base.BaseEstimator or list
            The model or list of models to evaluate.

        model_names: list, optional (Default=None)
            A list with the names of the models.

        Returns
        -------
        StreamModel or list
            The trained model(s).

        """
        self._init_evaluation(model=model, stream=stream, model_names=model_names)

        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model = self._train_and_test()

            if self.show_plot:
                self.visualizer.hold(image_name)

            return self.model
    
    def _init_plot(self):
        """ Initialize plot to display the evaluation results.
        """
        
        if self.show_plot:
            
            self.visualizer = FGTEvaluationVisualizer(task_type=self._task_type,
                                                   n_wait=self.n_sliding,
                                                   dataset_name=self.stream.get_data_info(),
                                                   metrics=self.metrics,
                                                   n_models=self.n_models,
                                                   model_names=self.model_names,
                                                   data_dict=self._data_dict)
            