# EC552 HW1
## Description
Tool for modifying Cello Input file for improved genetic circuit performance
## Authors
- Anirudh Watturkar
- Liam Murray

**Inputs:**
1. .v : Input Verilog file, representing an circuit.
2. .UCF.json : A JSON representation of the Library used by Cello, representing the chassis organism that the genetic circuit will exist within.
3. .input.json : A JSON representation of the Library used by Cello, representing the input signals into the genetic circuit will exist within.
4. .output.json : A JSON representation of the Library used by Cello, representing the output of the genetic circuit.

**Outputs:**
1. Score : The new score of the circuit (in the command line)
2. Delta : Difference between the new and old scores of the circuit (in the command line)
3. .NEWinput.json : The modified input file.

# Installation Instructions

## 1. Ensure python 3.8+ & pip are installed
## 2. Install Docker
## 3. Pull docker instance
    docker pull cidarlab/cello-dnacompiler:latest
## 4. Install requirements via pip:
    pip install -r requirements.txt
- Libs used:
  - numpy v1.20.2
  - pandas v1.2.3
  - celloapi2 v0.2.4 (has the following dependencies):
    - python-dateutil v2.8.1
    - pytz v2021.1
    - PyYAML v5.4.1
    - six v1.15.0
## 5. Ensure cello docker instance is running
## 6. Create File Structure for running Cello    
### There are two ways to do this:
    
  1. Clone the homework 1 template, then place cli.py, optimizeInput.py, and evaluate.py in the top-level folder of the repository. Then, create an output directory.
        
        ` git clone https://github.com/CIDARLAB/homework1-template`
        
        Your file structure should now be like this:

        ```
        ├── input
        │   ├── and.v
        │   ├── Eco1C1G1T1.input.json
        │   ├── Eco1C1G1T1.output.json
        │   ├── Eco1C1G1T1.UCF.json
        │   ├── nand.v
        │   ├── options.csv
        │   ├── struct.v
        │   └── xor.v
        ├── cli.py            <-
        ├── evaluate.py       <-
        ├── main.py           x
        ├── optimizeInput.py  <-
        ├── output
        ├── poetry.lock       x
        ├── pyproject.toml    x
        └── README.md         x
        ```
  2. Recreate the above file structure and populate the input directory with the input, output ucf, options.csv, and verilog files. Note: the files marked with 'x' next to them in the above structure do not need to be recreated if you are doing this manually.
# Running

## There are 3 ways to perform optimization & evaulate the results:

## 1. Using `cli.py`
#### Navigate to the directory where `cli.py` is found, then follow the instructions below:

### -> To load *input.json, optimize its paramters, and save to a new file:
    python cli.py -m <og_input_file.json> <new_file_name.json> <chassis_name>
    
    ex: 
    $ python cli.py -m Eco1C1G1T1.input.json Eco1C1G1T1NEW.input.json Eco1C1G1T1
### -> To run Cello with a given *input.json file:
    python cli.py -r <input.json> <num_input_signals> <chassis_name> <verilog_file.v>
    
    ex:
    $ python cli.py -r Eco1C1G1T1NEW.input.json 2 Eco1C1G1T1 and.v
### -> To compare the original input to the optimized one:
    python cli.py -mr <og_input_file.json> <new_file_name.json> <num_input_signals> <chassis_name> <verilog_file.v>
    
    ex:
    $ python cli.py -mr Eco1C1G1T1.input.json Eco1C1G1T1NEW.input.json 2 Eco1C1G1T1 and.v
### -> For full usage instructions:
    $ python cli.py -h
## 2. Using `evaluate.py`

### If you cloned the template repo, then you should be able to simply run
    $ python evaluate.py

## For custom input files to Cello make the following changes to `evaluate.py`:

### -> If you are using a chassis other than Eco1C1G1T1, modify the `chassis_name` variable in evaluate.py
### -> If the circuit requires other than 2 inputs, set `signal_input` equal to the number of inputs to the circuit
### -> If a custom verilog file is desired, set `v_file` to the new verilog file name
### -> If a different options file is used, changed `options` to the new options file name

### Notes:
- Make sure that there are folders named output and input, which contains the 4 input files listed above in the Inputs section.
- Detailed output returned from Cello will be stored in the output folder.
- The output score and delta values will be printed to the command line; the .NEWinput.json file will save to the input file.

## 3. With your own custom script

### If you have your own custom script and would like to integrate our optimization script into it, place `optimizeInput.py` in the same directory as your script, then add the following to the top of your script:

    from optimizeInput import *

### Then, to get the optimal input file via our optimization method, add the following in your script (replacing the filenames as necessary):
    input_class_list = get_input_models_in_class('Eco1C1G1T1.input.json')
    gate_class_list = get_file_gate_models_in_class('Eco1C1G1T1.UCF.json')
    mod_inputs_list = compute_optimal_parameters(input_class_list, gate_class_list)
    save_input_class_in_file(mod_inputs_list,'Eco1C1G1T1.input.json','Eco1C1G1T1.NEWinput.json' )