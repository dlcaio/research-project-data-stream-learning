import pandas as pd
from skmultiflow.data import FileStream
from skmultiflow.lazy import FGTKNN
from skmultiflow.evaluation import FGTEvaluatePrequential

"""
This is a python program to generate evaluation results for 4 datasets with forgetting feature.
The aim is to compare models with the same k value and maximum window size
that forget different number of samples every 1000 samples seen.
For that purpose there are 5 forgetting values for each model: 0, 100, 250, 500, 750
"""


"""
initializing streams by passing csv datasets to FileStream constructor
"""
#stream_chessboard = FileStream('fgt-knn-tests/chessboard/chessboard-dataset.csv')
#stream_rotating_hyperplane = FileStream('fgt-knn-tests/rotating-hyperplane/rotating-hyperplane-dataset.csv')
#stream_sea_concepts = FileStream('fgt-knn-tests/sea-concepts/sea-concepts-dataset.csv')
#stream_weather = FileStream('fgt-knn-tests/weather/weather-dataset.csv')
#stream_poker = FileStream('fgt-knn-tests/poker/poker.csv')
#stream_mixed = FileStream('fgt-knn-tests/mixed-drift/mixed-drift.csv')
#stream_interchanging = FileStream('fgt-knn-tests/interchanging-rbf/interchanging.csv')

"""
preparing each data stream for use
"""
#stream_chessboard.prepare_for_use()
#stream_rotating_hyperplane.prepare_for_use()
#stream_sea_concepts.prepare_for_use()
#stream_weather.prepare_for_use()
#stream_poker.prepare_for_use()
#stream_mixed.prepare_for_use()
#stream_interchanging.prepare_for_use()


def generate_knn_models(k, wind_size):
    # list with number of samples to be forgotten
    n_samples_fgt = [100, 250, 500, 750]
    # initialize list of knn's with model which won't have data forgotten
    knn_models = [FGTKNN(n_neighbors = k, max_window_size=wind_size, fgt = False)]
    # loop to add knn's that will have data forgotten
    for i in range(4):
        knn_models.append(FGTKNN(n_neighbors = k, max_window_size=wind_size, fgt_n = n_samples_fgt[i]))
    return knn_models

def generate_different_k_values_knn_models(starting_k_value):
    # list that will contain lists with knn models, each with its own 'k' values
    knn_models_nested = []
    # loop to append to knn_models_nested knn models with k = [3, 5]
    for j in range(starting_k_value, 6, 2):
        knn_models_nested.append(generate_knn_models(j, 5000)) 
    return knn_models_nested

def evaluate(dataset_name, stream, starting_k_value = 3):
    knn_list = generate_different_k_values_knn_models(starting_k_value)
    for i in range(len(knn_list)):
        
        file_name = 'results_k='+str(knn_list[i][0].n_neighbors)+'_ws='+str(knn_list[i][0].max_window_size)
        evaluator = FGTEvaluatePrequential(max_samples=2000000,
                                     show_plot=False,
                                     pretrain_size=knn_list[i][0].n_neighbors,
                                     n_wait=100,
                                     metrics=['accuracy'],
                                     output_file='fgt-knn-tests/' + dataset_name + '/' + file_name,
                                     fgt_freq=1000)
        
        evaluator.evaluate(stream = stream, model = knn_list[i], image_name='fgt-knn-tests/' + dataset_name + '/' + file_name, model_names = ['knn', 'knn100', 'knn250', 'knn500', 'knn750'])

"""
generate results for each dataset
"""
#evaluate('chessboard', stream_chessboard)
#evaluate('rotating-hyperplane', stream_rotating_hyperplane)
#evaluate('sea-concepts', stream_sea_concepts)
#evaluate('weather', stream_weather)
#evaluate('poker', stream_poker)
#evaluate('interchanging-rbf', stream_interchanging)
#evaluate('mixed-drift', stream_mixed)


