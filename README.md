# EC552_HW1

Tool for modifying User Constraints File for improved genetic circuit design

**Inputs:**
1. .v : Input Verilog file, representing an circuit.
2. .UCF.json : A JSON representation of the Library used by Cello, representing the chassis organism that the genetic circuit will exist within.
3. .input.json : A JSON representation of the Library used by Cello, representing the input signals into the genetic circuit will exist within.
4. .output.json : A JSON representation of the Library used by Cello, representing the output of the genetic circuit.

**Outputs:**
1. The Score : The new score of the circuit
2. The Delta : Difference between the new and old scores of the circuit
3. .NEWinput.json : The modified input file.

# Installing

If they are not already install, install the [celloapi2](https://github.com/CIDARLAB/celloapi2) and [NumPy](https://numpy.org/) modules.
To install this program, simply clone the repository to your directory of choice. 

# Running
To run the program, simply run main.py from the command line.
If you are using a chassis other than Eco1C1G1T1, modify the *chassis_name* variable in main.py.
If the circuit requires other than 2 inputs, set *signal_input* equal to the number of inputs to the circuit.

Make sure that there are folders named output and input, which contains the 4 input files listed above in the Inputs section.
The output score and delta values will be printed to the command line; the .NEWinput.json file will save to the input file.
