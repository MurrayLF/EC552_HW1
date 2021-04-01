import os
from optimizeInput import *
from itertools import combinations
from celloapi2 import CelloQuery, CelloResult

# Set our directory variables.
in_dir = os.path.join(os.getcwd(), 'input')
out_dir = os.path.join(os.getcwd(), 'output')

#Number of inputs into circuit
signal_input = 2

# Set our input files.
chassis_name = 'Eco1C1G1T1'
in_ucf = f'{chassis_name}.UCF.json'
v_file = 'and.v'
options = 'options.csv'
input_sensor_file = f'{chassis_name}.input.json'
output_device_file = f'{chassis_name}.output.json'

#Determine best score for unmodified input for calculation of delta
eval=True
if eval:
    del_best_score
    
    qdel= CelloQuery(
        input_directory=in_dir,
        output_directory=out_dir,
        verilog_file=v_file,
        compiler_options=options,
        input_ucf=in_ucf,
        input_sensors=input_sensor_file,
        output_device=output_device_file,
    )
    signals = qdel.get_input_signals()
    signal_pairing = list(combinations(signals, signal_input))
    for signal_set in signal_pairing:
        signal_set = list(signal_set)
        qdel.set_input_signals(signal_set)
        qdel.get_results()
        try:
            delres = CelloResult(results_dir=out_dir)
            if delres.circuit_score > del_best_score:
                del_best_score = delres.circuit_score
        except:
            pass
        qdel.reset_input_signals()

    qdel.get_results()
    delres = CelloResults(result_dir=out_dir)

#Perform calculations and modify input file
input_class_list = get_input_models_in_class(input_sensor_file)
gate_class_list = get_file_gate_models_in_class(in_ucf)
modified_input_class_list = compute_optimal_parameters(input_class_list,gate_class_list)
NEWinput_sensor_file = f'{chassis_name}.NEWinput.json'
save_input_class_in_file(modified_input_calss_list,input_sensor_file,NEWinput_sensor_file)

#Calculate best score for modified input
best_score=0

q = CelloQuery(
    input_directory=in_dir,
    output_directory=out_dir,
    verilog_file=v_file,
    compiler_options=options,
    input_ucf=in_ucf,
    input_sensors=NEWinput_sensor_file,
    output_device=output_device_file,
)
signals = q.get_input_signals()
signal_pairing = list(combinations(signals, signal_input))
for signal_set in signal_pairing:
    signal_set = list(signal_set)
    q.set_input_signals(signal_set)
    q.get_results()
    try:
        res = CelloResult(results_dir=out_dir)
        if res.circuit_score > best_score:
            best_score = res.circuit_score
            best_chassis = chassis
            best_input_signals = signal_set
    except:
        pass
    q.reset_input_signals()

#Output best score for modified input
print("New best score: ",best_score)

#Caclulate and output delta
if eval:
    delta = best_score - del_best_score    
    print("Delta: ",delta)
