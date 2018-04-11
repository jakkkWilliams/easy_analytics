import easy_analytics

# IMPLETENT REGRESSION BY DEEP LEARNING
# Hidden Layers: 3-3-3
# target_variable = b_0 + b_1*var1 + b_2*var2 + b_3*var3
run_one_loop(
    model_name = 'poisson_model1', # Used in result csv
    data = pd.read_csv('dataset.csv'),
    varlist = ['var1', 'var2', 'var3'],
    target = ['target_variable_name'],
    learn_ratio = 0.75, # Divide dataset into learn, test by specified ratio
    learning_rate = 0.0001, # Learning rate
    hidden_layers = [3, 3, 3], # Specify "[]" if not using deep learning
    nepoch = 500, # Number of epochs
    batchsize = 100, # Batchsize for learning process
    activation_function = 'relu', # Activation function for ALL THE LAYERS
    display_step = 100, # Display learning process by the specified freq
    result_with_no_step = False, # Hide learning process and show results only
    save_result = True, # Add result in [result_filename].csv
    result_filename = 'result' # Filename for the result data
)

# IMPLETENT POISSON REGRESSION
# target_variable = b_0 + b_1*var1 + b_2*var2 + b_3*var3
run_one_loop(
    model_name = 'poisson_model1', # Used in result csv
    data = pd.read_csv('dataset.csv'),
    varlist = ['var1', 'var2', 'var3'],
    target = ['target_variable_name'],
    learn_ratio = 0.75, # Divide dataset into learn, test by specified ratio
    learning_rate = 0.0001, # IGNORED
    hidden_layers = [], # Specify "[]" if not using deep learning
    nepoch = 500, # IGNORED
    batchsize = 100, # IGNORED
    activation_function = 'poisson', # Activation/Distribution
    display_step = 100, # Display learning process by the specified freq
    result_with_no_step = False, # Hide learning process and show results only
    save_result = True, # Add result in [result_filename].csv
    result_filename = 'result' # Filename for the result data
)