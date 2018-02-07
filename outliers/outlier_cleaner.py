#!/usr/bin/python
import numpy


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    cleaned_data = range(len(predictions))
    for idx, val in enumerate(cleaned_data):
        cleaned_data[idx] = [ages[idx], net_worths[idx], abs(predictions[idx][0] - net_worths[idx][0])]

    for index in range(0, 9):
        max_error = 0.0
        index_remove = 0
        for idx, val in enumerate(cleaned_data):
            if (max_error < cleaned_data[idx][2]):
                max_error = cleaned_data[idx][2]
                index_remove = idx
        del cleaned_data[index_remove]

    return cleaned_data

