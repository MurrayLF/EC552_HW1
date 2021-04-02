"""
EC552 HW 1-Anirudh Watturkar, Liam Murray

Functions necessary to interpret, optimize, and write an input file for cello.

This module contains classes to represent biological repressors and input
promoters, provides tools to parse input and ucf files containing this
formation, provides a function to optimize the input file, and provides a
function to write the modified inputs to a new file.


Example usage instructions
--------------------------

# (1) Get input classes from input file
input_class_list = get_input_models_in_class('Eco1C1G1T1.input.json')

# (2) Get repressor(gate) classes from UCF file
gate_class_list = get_file_gate_models_in_class('Eco1C1G1T1.UCF.json')

# (3) Perform optimization on inputs
mod_inputs_list = compute_optimal_parameters(input_class_list, gate_class_list)

# (4) Write modifications to new input file
save_input_class_in_file(mod_inputs_list,'Eco1C1G1T1.input.json','Eco1C1G1T1.NEWinput.json' )
"""

import json
import copy
import math
from shutil import copyfile
import os

import pandas as pd
import numpy as np

class Repressor:
    """
    Contains the parameters and output functions to represent a biological repressor.

    The class provides methods to get the output of the given repressor when implemented as both
    a NOT and NOR type gate. This class also provides methods to get the gradient with respect to the
    input scaling factor(s) for NOT and NOR implementations.

    Attributes:
        name: Name of the repressor

        ymax: The output value when expressing digital 1

        ymin: The output value when expressing digital 0

        K: The K_d of the response function

        n: The slope of the response function
    """

    def __init__(self, name, ymax, ymin, K, n):
        """Init Repressor with its name, ymax, ymin, the K_d, and the slope factor."""
        self.name = name
        self.ymax = ymax
        self.ymin = ymin
        self.K = K
        self.n = n

    def outputNOT(self, input_obj, truth_table=False, log_score=True):
        """
        Computes NOT gate output and score given input.

        Computes NOT gate output using hill response function equation and the
        given Input object.

        Parameters
        ----------
        input_obj : Input object
            Input object representing the input promoter connecting to the gate.
        truth_table : Bool, optional
            Flag to return the truth table instead of score. The default is False.
        log_score : TYPE, optional
            Flag to set the score to log10 scale. The default is True.

        Returns
        -------
        **If truth_table is set to True:
        Dict
            A Dict that containts the following:

            Key 'tt': A list containing the output states of the NOT gate for each input state.
            The zeroth index corresponds to the input at the low state, and the first
            corresponds to the input at the high state.

            Key 'score': The computed score of the NOT gate given the input, stored as a float.

        **If truth_table is set to False:
        Float
            The computed score of the NOT gate given the input.
        """

        # Initializes truth table list
        tt = [0,0]
        # Generates the gate output for each input logic level
        tt[0] = self.ymin+((self.ymax-self.ymin)/(1+((input_obj.output(0))/self.K)**self.n))
        tt[1] = self.ymin+((self.ymax-self.ymin)/(1+((input_obj.output(1))/self.K)**self.n))

        if log_score:
            score = math.log(tt[0]/tt[1],10)
        else:
            score = tt[0]/tt[1]

        if truth_table:
            return {'truth_table':tt, 'score':score}
        else:
            return score

    def outputNOR(self, input_obj1, input_obj2, truth_table=False, log_score=True):
        """
        Computes NOR gate output and score given the inputs.

        Computes NOR gate output using hill response function equation and the
        given Input objects.


        Parameters
        ----------
        input_obj1 : Input object
            First Input object representing the input promoter connecting to the gate.
        input_obj2 : Input object
            Second Input object representing the input promoter connecting to the gate.
        truth_table : Bool, optional
            Flag to return the truth table instead of score. The default is False.
        log_score : TYPE, optional
            Flag to set the score to log10 scale. The default is True.

        Returns
        -------
        **If truth_table is set to True:
        List
            A list containing the output states of the NOR gate for each input state.
            The decimal value of the index represents the binary value of the input.
            Ex: tt[2] -> NOR output when input_obj1 is high and input_obj is low, or binary '10'

        **If truth_table is set to False:
        Float
            The computed score of the NOR gate given the input.

        """
        # Iniitalizes the truth table list
        tt = [0,0,0,0]
        # Generates the gate output for each input level pair
        tt[0b00] = self.ymin+((self.ymax-self.ymin)/(1+((input_obj1.output(0)+input_obj2.output(0))/self.K)**self.n))
        tt[0b01] = self.ymin+((self.ymax-self.ymin)/(1+((input_obj1.output(0)+input_obj2.output(1))/self.K)**self.n))
        tt[0b10] = self.ymin+((self.ymax-self.ymin)/(1+((input_obj1.output(1)+input_obj2.output(0))/self.K)**self.n))
        tt[0b11] = self.ymin+((self.ymax-self.ymin)/(1+((input_obj1.output(1)+input_obj2.output(1))/self.K)**self.n))

        if log_score:
            score = math.log(tt[0]/max(tt[1],tt[2],tt[3]), 10)
        else:
            score = tt[0]/max(tt[1],tt[2],tt[3])

        if truth_table:
            return tt
        else:
            return score

    def gradNOT(self, input_obj, factor):
        """
        Calculates the gradient of the NOT gate wrt to the scaling parameter
        on the input object's logic levels.

        Computes the gradient of the NOT score function, with respect to
        a parameter scaling the input to the function. The gradient is represented
        by the log ratio of the gate's hill response function at the low input over
        the gate's hill response function at the high input. The scaling parameter represents
        scaling the input promoter via stronger_promoter(a) or weaker_promoter(a),
        where a is the scaling parameter.

        Parameters
        ----------
        input_obj : Input object
            Input object representing the input promoter connecting to the gate.
        factor : Float
            Scaling parameter to be evaluated in the gradient function.

        Returns
        -------
        d_score_d_factor : Float
            The gradient of the score function evaluated with the given factor input.

        """

        # Ensures the input object outputs its default low and high values
        input_obj.reset()

        # Components whose product make up the numerator of the gradient expression
        numert1= -self.n*(self.ymax - self.ymin)
        numert2 = (((input_obj.output(1)*factor)/self.K)**self.n) - (((input_obj.output(0)*factor)/self.K)**self.n)
        numert3 = self.ymin*(((input_obj.output(1)*factor)/self.K)**self.n)*(((input_obj.output(0)*factor)/self.K)**self.n) - self.ymax

        # Components whose product make up the denominator of the gradient expression
        denomt1 = math.log(10)*factor
        denomt2 = (((input_obj.output(0)*factor)/self.K)**self.n) + 1
        denomt3 = self.ymin*(((input_obj.output(0)*factor)/self.K)**self.n) + self.ymax
        denomt4 = (((input_obj.output(1)*factor)/self.K)**self.n) + 1
        denomt5 = self.ymin*(((input_obj.output(1)*factor)/self.K)**self.n) + self.ymax

        d_score_d_factor = (numert1*numert2*numert3)/(denomt1*denomt2*denomt3*denomt4*denomt5)

        return d_score_d_factor

    def bestFactorNOT(self, input_obj):
        """
        Calculates the best input scaling factor for the current NOT gate.

        Uses the resulting expression from solving for the scaling factor when
        the gradient of the gate's NOT score function is set to zero to get
        the best input scaling factor for the current gate.

        Parameters
        ----------
        input_obj : Input object
            Input object representing the input promoter connecting to the gate.

        Returns
        -------
        factor : Float
            Optimal scaling parameter for the current gate.

        """

        # Ensures the input object outputs its default low and high values
        input_obj.reset()

        input_low = input_obj.output(0)
        input_high = input_obj.output(1)

        # Explicitly solves for the scaling paramter when the gradient is set to zero
        factor = (self.ymax/self.ymin)**(1/(2*self.n)) * (self.K/math.sqrt(input_low*input_high))

        return factor

    def gradNOR(self, input_obj1, input_obj2, factor_list):
        """
        Calculates the gradient of the NOR gate wrt to the scaling parameters
        on the each of the input objects' logic levels.

        Computes the gradient of the NOR score function, with respect to
        the parameters scaling the inputs to the function. The gradient is represented
        by the log ratio of the gate's hill response function when both inputs are low over
        the max gate's hill response function among each of the three remaining input pairs.
        The scaling parameters represent scaling the input promoters via
        stronger_promoter(a) or weaker_promoter(a), where a is the scaling parameter.


        Parameters
        ----------
        input_obj1 : Input object
           Input object representing the input promoter connecting to the gate.
        input_obj2 : Input object
            Input object representing the input promoter connecting to the gate.
        factor_list : List
            List representing the two scaling parameters. The first entry corresponds
            to input_obj1 and the second input_obj2

        Returns
        -------
        Numpy Array
            Numpy array containing the outputted gradients wrt each input's
            scaling parameter. The first entry in the array represents the
            score function's gradient wrt the first input's scaling parameter.
            The second entry in the array represents the score function's gradient
            wrt the second input's scaling parameter.
        """

        # Ensures both inputs output their default values
        input_obj1.reset()
        input_obj2.reset()

        # Converts a scaling factor < 1 to the domain of the weaker promoter operation
        # for each factor
        if factor_list[0] < 1:
            input_obj1.weak_p(1/factor_list[0])
        else:
            input_obj1.strong_p(factor_list[0])
        if factor_list[1] < 1:
            input_obj2.weak_p(1/factor_list[1])
        else:
            input_obj2.strong_p(factor_list[1])

        # Finds the input logic level pair that corresponds to the max
        # output value for a logical low level for the circuit.
        # val1 and val2 represent the digital output level of
        # input_obj1 and input_obj2, respectively

        tt = self.outputNOR(input_obj1, input_obj2, truth_table=True)
        max_off = tt.index(max(tt[1:]),1)

        if max_off == 1:
            val1 = 0
            val2 = 1
        elif max_off == 2:
            val1 = 1
            val2 = 0
        elif max_off == 3:
            val1 = 1
            val2 = 1

        # Stores correpsonding input values for the gate's minimum 'TRUE'
        # and maximum 'FALSE' values.

        min_on_in1 = input_obj1.output(0)
        min_on_in2 = input_obj2.output(0)

        max_off_in1 = input_obj1.output(val1)
        max_off_in2 = input_obj2.output(val2)


        # Computes the gradient wrt the first scaling parameter

        # Computes the numerator expression for the first fraction in the gradient
        numer1 = max_off_in1 * self.n * (self.ymax - self.ymin) * ((max_off_in1*factor_list[0] + max_off_in2*factor_list[1])/self.K)**self.n
        # Computes the denominator terms for the first fraction in the gradient
        denom1_t1 = math.log(10)*max_off_in1*factor_list[0] + max_off_in2*factor_list[1]
        denom1_t2 = ((((max_off_in1*factor_list[0] + max_off_in2*factor_list[1])/self.K)**self.n) + 1) ** 2
        denom1_t3 = ((self.ymax - self.ymin)/(((((max_off_in1*factor_list[0]+ max_off_in2*factor_list[1])/self.K)**self.n) + 1))) + self.ymin

        frac1 = numer1/(denom1_t1*denom1_t2*denom1_t3)
        # Computes the numerator expression for the first fraction in the gradient
        numer2 = min_on_in1 * self.n * (self.ymax - self.ymin) * ((min_on_in1*factor_list[0]+ min_on_in2*factor_list[1])/self.K)**self.n
        # Computes the denominator terms for the first fraction in the gradient
        denom2_t1 = math.log(10)*min_on_in1*factor_list[0] + min_on_in2*factor_list[1]
        denom2_t2 = ((((min_on_in1*factor_list[0]+ min_on_in2*factor_list[1])/self.K)**self.n) + 1) ** 2
        denom2_t3 = ((self.ymax - self.ymin)/(((((min_on_in1*factor_list[0]+ min_on_in2*factor_list[1])/self.K)**self.n) + 1))) + self.ymin

        frac2 = numer2/(denom2_t1*denom2_t2*denom2_t3)

        d_score_d_factor1 = frac1-frac2

        # Computes the gradient wrt the first scaling parameter. The denominator
        # terms for each fraction are the same as previous gradient

        # Computes the numerator expression for the first fraction in the gradient
        numer1 = max_off_in2 * self.n * (self.ymax - self.ymin) * ((max_off_in1*factor_list[0]+ max_off_in2*factor_list[1])/self.K)**self.n
        # Computes the numerator expression for the first fraction in the gradient
        numer2 = min_on_in2 * self.n * (self.ymax - self.ymin) * ((min_on_in1*factor_list[0] + min_on_in2*factor_list[1])/self.K)**self.n

        frac1 = numer1/(denom1_t1*denom1_t2*denom1_t3)
        frac2 = numer2/(denom2_t1*denom2_t2*denom2_t3)

        d_score_d_factor2 = frac1-frac2

        return np.array([d_score_d_factor1, d_score_d_factor2])


class Input:
    """
    Contains the parameters to represent an input promoter.

    The class provides methods to get the output of the given input, as well as
    modify the input through DNA and protein engineering operations.

    Attributes
    ----------
        name: Name of the input

        ymax: The output value when expressing digital 1

        ymin: The output value when expressing digital 0

        __ymax_og: The initial value of ymax. Remains unchanged after instantiation

        __ymin_og: The initial value of ymin. Remains unchanged after instantiation

    """

    def __init__(self,name, ymax, ymin):
        """Inits Input with its name, ymax, and ymin. Copies ymax and ymin in private variables"""
        self.name = name
        self.ymax = ymax
        self.ymin = ymin
        self.__ymax_og = ymax
        self.__ymin_og = ymin

    def output(self, logic_level):
        """Provides the real-value output at the given logic level. 0 -> FALSE, 1-> TRUE"""
        if logic_level:
            return self.ymax
        else:
            return self.ymin

    def reset(self):
        """Resets the internal ymin and ymax to their orignal values when instantiated"""
        self.ymax = self.__ymax_og
        self.ymin = self.__ymin_og

    def strong_p(self,x):
        """Perform the stronger promoter operation with the given factor 1<=factor"""
        if x>=1:
            self.ymax = self.__ymax_og * x
            self.ymin = self.__ymin_og * x
        else:
            print("Invalid stronger promoter factor")

    def weak_p(self,x):
        """Perform the weaker promoter operation with the given factor 1<=factor"""
        if x >= 1:
            self.ymax = self.__ymax_og / x
            self.ymin = self.__ymin_og /x
        else:
            print("Invalid weaker promoter factor")

    def stretch(self,x):
        """Perfom the stretch operation with the given factor. 1<=factor<=1.5"""
        if x <= 1.5 and x >= 1:
            self.ymax = self.__ymax_og * x
            self.ymin = self.__ymin_og / x
        else:
            print("Invalid stretch factor")


def get_input_models_in_class(filename):
    """
    Creates a list of Input instances containing the logic paramters
    for each input from a given JSON input file.

    Parameters
    ----------
    filename : Path
        Path to your *input.json file. Can be relative or absolute.

    Returns
    -------
    input_class_list : List of Input objects
        List of objects representing the parameters and operations of each input
    """

    input_class_list = []

    # Converts the input file into a list of dicts representing the JSON file
    with open(filename, "r") as f:
        data = json.load(f)

    # Searches for the name, ymin, and ymax of a given input and creates an
    # Input class instance with those paramters. These classes are then stored in a list.
    for collection in data:
        if collection['collection'] == 'models':
            params = collection['parameters']
            for param in params:
                if param['name'] == 'ymax':
                    ymax = param['value']
                if param['name'] == 'ymin':
                    ymin = param['value']
            input_inst = Input(collection['name'].replace('_sensor_model',''), ymax, ymin)
            input_class_list.append(input_inst)
    f.close()
    return input_class_list

def get_file_gate_models_in_class(filename):
    """
    Creates a list of Repressor instances containing the response function paramters
    for each gate from a given JSON UCF file

    Parameters
    ----------
    filename : string
        Filename of the JSON UCF File

    Returns
    -------
    gate_class_list : list
        List of Repressor instances containing the response function parameters
        of each of the gates

    """
    # Opens JSON UCF file and isolates models collections
    with open(filename,"r") as f:
        data = json.load(f)
    model_list = list(filter(lambda model: model['collection'] == 'models', data))

    gate_class_list = []
    # Creates a list of Repressor instances with a given gate's name and response function paramters
    for model in model_list:
        for param in model['parameters']:
            if param['name'] == 'ymax':
                ymax = param['value']
            if param['name'] == 'ymin':
                ymin = param['value']
            if param['name'] == 'K':
                K = param['value']
            if param['name'] == 'n':
                n = param['value']
        rep = Repressor(model['name'].replace('_model',''),ymax,ymin,K,n)
        gate_class_list.append(rep)

    f.close()
    return gate_class_list

def save_input_class_in_file(input_class_list, original_input_file, new_input_file):
    """
    Stores a list of Input class instances in the format of a cello-readable
    input.json file.

    Creates a file by copying a reference file, then modifying the file copy with
    the given Input class list's parameters.

    Parameters
    ----------
    input_class_list : List of Input objects
        List of Input objects representing the input promoters.
    original_input_file : Path string
        Filename/path of the original input file.
    new_input_file : Path string
        Filename/path of the newly created input file.

    Returns
    -------
    None.

    """
    # Converts relative paths to absolute paths
    if not os.path.isabs(original_input_file):
        current_path = os.getcwd()
        original_input_file = os.path.join(current_path, original_input_file)
    if not os.path.isabs(new_input_file):
        current_path = os.getcwd()
        new_input_file = os.path.join(current_path, new_input_file)

    # Creates copy of reference file
    copyfile(original_input_file, new_input_file)

    # Creates JSON structure from new file
    with open(new_input_file, "r") as f:
        data = json.load(f)
    f.close()

    # Finds the corresponding input models, and replaces their parameters
    # with those in the Input class list
    name_list = [input_obj.name for input_obj in input_class_list]
    for i,obj in enumerate(data):
        if obj['collection'] == "models":
            name = obj['name'].replace('_sensor_model','')
            curr_input = input_class_list[name_list.index(name)]
            for j,param in enumerate(data[i]['parameters']):
                if param['name'] == "ymax":
                    data[i]['parameters'][j]['value'] = curr_input.ymax
                if param['name'] == "ymin":
                    data[i]['parameters'][j]['value'] = curr_input.ymin

    # Writes modifications to specified file
    with open(new_input_file,"w") as f:
        json.dump(data, f, indent="\t")
    f.close()

def get_avg_scoreNOT(input_obj, gate_class_list):
    """Computes the average score for a given input across all NOT gates"""
    score_list = []
    for gate in gate_class_list:
        score = gate.outputNOT(input_obj)
        score_list.append(score)
    score_list = np.array(score_list)
    return score_list.mean()

def get_avg_scoreNOR(input_obj1,input_obj2, gate_class_list):
    """Computes the average score for a given pair of inputs across all NOR gates"""
    score_list = []
    for gate in gate_class_list:
        score = gate.outputNOR(input_obj1, input_obj2)
        score_list.append(score)
    score_list = np.array(score_list)

    return score_list.mean()

def compute_optimal_parameters(input_class_list, gate_class_list):
    """
    Calculates the optimal input scaling parameters through a gradient ascent-based
    score maximization algorithm.

    The score is modeled as a weighted average of the average score across all
    gates for a given input passed into a NOT gate, and a NOR gate with all of
    the other inputs. The gradient with respect to the input's scaling parameter
    is then computed, and this value is maximized wrt average score
    via gradient ascent.

    Parameters
    ----------
    input_class_list : List of Input objects
        List of Input objects representing the input promoters.
    gate_class_list : List of Repressor objects
        List of Repressor objects representing the biological gates available.

    Returns
    -------
    input_mod_list : List of Input objects
        Modifed list of Input objects that has the optimal scaling to the input's parameters.

    """

    # Creates a separate list to be modified once optimal paramters are calculated
    input_mod_list = copy.deepcopy(input_class_list)

    # Creates an empty data frame to store the results of optimzation
    df = pd.DataFrame(columns=['input','factor', 'not_score','delta_not','nor_score','delta_nor'])

    # Sets the weights for the weighted average score calculation based on the total number of inputs
    NOT_weight = 1/len(input_class_list)
    NOR_weight = (1-NOT_weight)/(len(input_class_list)-1)

    for input_obj in input_class_list:
        # Gets the other inputs in the list for computing the NOR gradient later
        input_obj_loc = input_class_list.index(input_obj)
        remaining_inputs = copy.deepcopy(input_class_list[:input_obj_loc] + input_class_list[input_obj_loc+1:])
        # Sets the inital scaling paramter. This is the paramter to be optimized
        factor = 0.01
        input_obj.reset()
        # Stores previous factor to determine if optimization is completed
        prev_factor = None
        ## Gradient ascent with 1000 iterations
        for i in range(0,1000):
            grad_NOT_list = []
            grad_NOR_list = []
            # Creates list of gradients for each gate
            for gate in gate_class_list:
                grad_NOT_list.append(gate.gradNOT(input_obj,factor))
                nor_list = []
                for inp in remaining_inputs:
                    nor_list.append(gate.gradNOR(input_obj, inp, [factor, 1])[0])
                # nor_list.append(gate.gradNOR(input_obj, input_obj,[factor,factor])[0])
                grad_NOR_list.append(nor_list)
            # Averages the gradient lists across all gates
            grad_NOR_list = np.array(grad_NOR_list)
            grad_NOT = np.array(grad_NOT_list).mean()
            grad_NOR_avgs = grad_NOR_list.mean(axis=0)
            # Computes the weighted average of gradient averages
            weighted_grad = 0
            for avg in grad_NOR_avgs:
                weighted_grad = weighted_grad + NOR_weight*avg
            weighted_grad = weighted_grad + NOT_weight*grad_NOT
            # Saves previous factor before updating the factor
            prev_factor = factor
            # Updates factor, and ensures factor is positive
            factor = factor + 0.51*weighted_grad
            if factor <= 0:
                factor = 0.00000001
            # Breaks if no improvement made to factor
            if prev_factor == factor:
                break
        ## Evaluate calculated best factor
        best_factor = factor
        input_obj.reset()
        # Computes baseline score for NOT and NOR gates
        baseline_NOT = get_avg_scoreNOT(input_obj, gate_class_list)
        baseline_NOR_list = []
        for inp in remaining_inputs:
            inp.reset()
            baseline_NOR_list.append(get_avg_scoreNOR(input_obj, inp, gate_class_list))
        baseline_NOR = np.array(baseline_NOR_list).mean()
        if best_factor < 1:
            input_obj.weak_p(1/best_factor)
        else:
            input_obj.strong_p(best_factor)
        # Computes new score for NOT and NOR gates
        avg_NOT_score = get_avg_scoreNOT(input_obj, gate_class_list)
        avg_NOR_score_list = []
        for inp in remaining_inputs:
            avg_NOR_score_list.append(get_avg_scoreNOR(input_obj, inp, gate_class_list))
        avg_NOR_score = np.array(avg_NOR_score_list).mean()
        # Appends results in dataframe
        df_entry = [[input_obj.name, best_factor, avg_NOT_score,avg_NOT_score-baseline_NOT, avg_NOR_score,avg_NOR_score-baseline_NOR]]
        tdf = pd.DataFrame(df_entry, columns=['input','factor', 'not_score','delta_not','nor_score','delta_nor'])
        df = df.append(tdf)
    # Generate modified input class list
    for i,input_obj in enumerate(input_mod_list):
        input_series = df.loc[df.input == input_obj.name]
        # If optimization results in a worse average NOT or NOR score, stretch instead
        if input_series.delta_not.values[0] < 0 or input_series.delta_nor.values[0] < 0:
            input_mod_list[i].stretch(1.5)
        else:
            factor = input_series.factor.values[0]
            if factor < 1:
                input_mod_list[i].weak_p(1/factor)
            else:
                input_mod_list[i].strong_p(factor)
    return input_mod_list
