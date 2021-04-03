import os
import sys
from itertools import combinations

from optimizeInput import *
from celloapi2 import CelloQuery, CelloResult

def print_help_menu():
    print("Use case: Generate optimized input file")
    print("python cli.py -m <original_input.json> <optimized_input.json> <chassis_name>")
    print("Parameters:")
    print("\t <original_input.json>: (String) The name of the original input file containing paramters to be optimized")
    print("\t <optimized_input.json>: (String) The name to give to the optimized input file")
    print("\t <chassis_name>: (String) Name of organism")
    print("Notes:")
    print("Please be sure that *.input.json is found in your cello input directory")    
    print("-------------------------\n")
    
    print("Use case: Run cello with set input file")
    print("python cli.py -r <input.json> <number of inputs> <chassis_name> <verilog_file.v>")
    print("Parameters:")
    print("\t <input.json>: (String) Input file to run Cello with")
    print("\t <number_of_inputs>: (Int, 0<n<=#Inputs in input file ) Number of input signals to circuit")
    print("\t <chassis_name>: (String) Name of organism")
    print("\t <verilog_file.v>: (String) Name of verilog file")
    print("Notes:")
    print("Please be sure that *.input.json, *.UCF.json, *.output.json, options.csv, and the specified verilog file are in your cello input directory")
    print("-------------------------\n")
    
    print("Use case: Run Cello on original input file, then optimize input and re-run. Displays score delta.")
    print("python cli.py [-rm/-mr] <original_input.json> <optimized_input.json> <number of inputs> <chassis_name> <verilog_file.v>")
    print("Parameters:")
    print("\t <original_input.json>: (String) The name of the original input file containing paramters to be optimized")
    print("\t <optimized_input.json>: (String) The name to give to the optimized input file")
    print("\t <number_of_inputs>: (Int, 0<n<=#Inputs in input file ) Number of input signals to circuit")
    print("\t <chassis_name>: (String) Name of organism")
    print("\t <verilog_file.v>: (String) Name of verilog file")
    print("Notes:")
    print("Please be sure that *.input.json, *.UCF.json, *.output.json, options.csv, and the specified verilog file are in your cello input directory")  

def print_m_help():
    print("Error: improper call to modify input file\n")
    print("Usage: python cli.py -m <original_input.json> <optimized_input.json> <chassis_name>")
    print("Parameters:")
    print("\t <original_input.json>: (String) The name of the original input file containing paramters to be optimized")
    print("\t <optimized_input.json>: (String) The name to give to the optimized input file")
    print("\t <chassis_name>: (String) Name of organism")
    print("Notes:")
    print("Please be sure that *.input.json is found in your cello input directory")    

def print_r_help():
    print("Error: not enough arguments given to run Cello\n")
    print("Usage: python cli.py -r <input.json> <number of inputs> <chassis_name> <verilog_file.v>")
    print("Parameters:")
    print("\t <input.json>: (String) Input file to run Cello with")
    print("\t <number_of_inputs>: (Int, 0<n<=#Inputs in input file ) Number of input signals to circuit")
    print("\t <chassis_name>: (String) Name of organism")
    print("\t <verilog_file.v>: (String) Name of verilog file")
    print("Notes:")
    print("Please be sure that *.input.json, *.UCF.json, *.output.json, options.csv, and the specified verilog file are in your cello input directory")

def print_rm_help():
    print("Error: not enough arguments given to run Cello\n")
    print("Usage: python cli.py [-rm/-mr] <original_input.json> <optimized_input.json> <number of inputs> <chassis_name> <verilog_file.v>")
    print("Parameters:")
    print("\t <original_input.json>: (String) The name of the original input file containing paramters to be optimized")
    print("\t <optimized_input.json>: (String) The name to give to the optimized input file")
    print("\t <number_of_inputs>: (Int, 0<n<=#Inputs in input file ) Number of input signals to circuit")
    print("Notes:")
    print("Please be sure that *.input.json, *.UCF.json, *.output.json, options.csv, and the specified verilog file are in your cello input directory")

def main():
    # Number of cmd args
    n = len(sys.argv)
    if n == 1:
        print("Not enough arguments supplied. Use 'python cli.py -h' for usage help")
        return
    if n > 1: 
        # Help option
        if sys.argv[1] == '-h':
                print_help_menu()
                return
        # Modify input file
        elif sys.argv[1] == '-m':
            if (n-2) != 3:
                print_m_help()
                return
            else:
                
                # Parse cmdline args for file names/chassis name
                chassis_name = sys.argv[4]
                
                in_dir = os.path.join(os.getcwd(), 'input')
                input_file_path = os.path.join(in_dir, sys.argv[sys.argv.index("-m")+1])
                ucf_file_path = os.path.join(in_dir, f'{chassis_name}.UCF.json')
                
                input_class_list = get_input_models_in_class(input_file_path)
                gate_class_list = get_file_gate_models_in_class(ucf_file_path)
                # Run optimization
                modified_input_class_list = compute_optimal_parameters(input_class_list,gate_class_list)
                # Save optimized file
                NEWinput_sensor_file = sys.argv[sys.argv.index("-m")+2]
                NEWinput_sensor_file_path = os.path.join(in_dir, NEWinput_sensor_file)
                save_input_class_in_file(modified_input_class_list,input_file_path,NEWinput_sensor_file_path)
                return
        # Run cello with set input file
        elif sys.argv[1] == "-r":
            if (n-2) != 4 :
                print_r_help()
                return
            else:
                # Set our directory variables.
                in_dir = os.path.join(os.getcwd(), 'input')
                out_dir = os.path.join(os.getcwd(), 'output')

                #Number of inputs into circuit
                signal_input = int(sys.argv[3])

                # Set our input files.
                chassis_name = sys.argv[4]
                in_ucf = f'{chassis_name}.UCF.json'
                v_file = sys.argv[5]
                options = 'options.csv'
                input_sensor_file = sys.argv[2]
                output_device_file = f'{chassis_name}.output.json'
                del_best_score = 0
                
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
                delres = CelloResult(results_dir=out_dir)
                print("Best circuit score:", del_best_score)
                return
        # Modify input file, run cello, and compare baseline to modified runs
        elif sys.argv[1] == "-mr" or sys.argv[1] == "-rm":
            if (n-2) != 5:
               print_rm_help()
               return
            else:
                # Set our directory variables.
                in_dir = os.path.join(os.getcwd(), 'input')
                out_dir = os.path.join(os.getcwd(), 'output')

                #Number of inputs into circuit
                signal_input = int(sys.argv[4])

                # Set our input files.
                chassis_name = sys.argv[5]
                in_ucf = f'{chassis_name}.UCF.json'
                v_file = sys.argv[6]
                options = 'options.csv'
                input_sensor_file = sys.argv[2]
                output_device_file = f'{chassis_name}.output.json'
                del_best_score = 0
                
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
                delres = CelloResult(results_dir=out_dir)

                in_dir = os.path.join(os.getcwd(), 'input')

                input_file_path = os.path.join(in_dir, input_sensor_file)
                ucf_file_path = os.path.join(in_dir, f'{chassis_name}.UCF.json')
                
                input_class_list = get_input_models_in_class(input_file_path)
                gate_class_list = get_file_gate_models_in_class(ucf_file_path)
                
                modified_input_class_list = compute_optimal_parameters(input_class_list,gate_class_list)
                
                NEWinput_sensor_file = sys.argv[3]
                NEWinput_sensor_file_path = os.path.join(in_dir, NEWinput_sensor_file)
                save_input_class_in_file(modified_input_class_list,input_file_path,NEWinput_sensor_file_path)

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
                delta = best_score - del_best_score    
                print("Delta: ",delta)

        else:
            print("Unknown command. Use 'python cli.py -h' for usage help")
            return
        
if __name__ == "__main__":
    main()