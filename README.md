# EC552_HW1

Tool for modifying User Constraints File for improved genetic circuit design

Inputs:
1. .v : Input Verilog file, representing an circuit.
2. .UCF.json : A JSON representation of the Library used by Cello, representing the chassis organism that the genetic circuit will exist within.
3. .input.json : A JSON representation of the Library used by Cello, representing the input signals into the genetic circuit will exist within.
4. .output.json : A JSON representation of the Library used by Cello, representing the output of the genetic circuit.

Outputs:
1. The Score : The new score of the circuit
2. *.UCF.json : The modified UCF file.


If they are not already install, install the pandas and NumPy packages for python.
