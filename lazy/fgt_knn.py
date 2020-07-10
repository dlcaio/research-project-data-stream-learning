from skmultiflow.lazy import KNN


import random
import numpy as numpy

class FGTKNN(KNN):
    
    def __init__(self,
                 fgt=True,
                 fgt_n_instances=100,
                 fgt_from_sub_set_length=1000,
                 n_neighbors=5,
                 max_window_size=1000,
                 leaf_size=30,
                 nominal_attributes=None):
        super().__init__(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         nominal_attributes=nominal_attributes)
        self.fgt = fgt
        self.fgt_n_instances = fgt_n_instances
        self.fgt_from_sub_set_length = fgt_from_sub_set_length

    def delete_element_at_index(self, i):
        """
        Delete element at a given index i from the sample window.
        """
        self.window._n_samples -= 1
        self.window._buffer = numpy.concatenate((self.window._buffer[:i, :], self.window._buffer[i + 1:, :]))
        
    def get_last_random(self, n_samples, sub_set_length):
        """
        get 'n_samples' randomly from the newest 'sub_set_length' elements
        """
        window_length = self.window.n_samples
        last_random_instances = []
        last_samples_starting_position = window_length - sub_set_length
        last_samples_range = range(last_samples_starting_position, window_length)        
        random_indexes = (random.sample(last_samples_range, n_samples))
        for i in range(n_samples):
            index = random_indexes[i]            
            random_instance = (self.window.buffer[index])
            last_random_instances.append(random_instance)
        return last_random_instances

    def delete_by_instance(self, instances):
        """
        looks for 'instances' given in the window and deletes them
        """
        for i in range(len(instances)):
            for j in range(len(self.window._buffer) - 1, -1, -1): 
                if(numpy.array_equal(instances[i], self.window._buffer[j])):
                    self.delete_element_at_index(j)
                    break
                
    def forget_last_random(self):
        """
        deletes newest random instances
        """
        instances = self.get_last_random(self.fgt_n_instances, self.fgt_from_sub_set_length)
        self.delete_by_instance(instances)
        
