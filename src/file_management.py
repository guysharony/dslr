import pickle

def save_parameters_to_file(parameters, filename):
    """
    Save a dictionary of parameters to a binary file

    Args:
        parameters (dict): dictionary of parameters to be saved
        filename (str): name of the file to save the parameters 
    """
    with open(filename, 'wb') as file:
        pickle.dump(parameters, file)
    print(f"Parameters saved to {filename}")

def load_parameters_from_file(filename):
    """
    Load parameters from a file using pickle.

    Parameters:
    - filename: The name of the file containing the parameters.

    Returns:
    - parameters: The loaded parameters.
    """
    try:
        with open(filename, 'rb') as file:
            parameters = pickle.load(file)
            print(f"Parameters loaded from the trained model")
            return parameters
    except FileNotFoundError:
        print(f"Parameters '{filename}' not loaded. Model hasn't been trained yet")
        return None