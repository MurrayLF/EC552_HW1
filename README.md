# EC552_HW1

Tool for modifying User Constraints File for improved genetic circuit design

Required Inputs
1. and.v : Input Verilog file, representing an AND circuit.
2. Eco1C1G1T1.UCF.json : A JSON representation of the Library used by Cello, representing the chassis organism that the genetic circuit will exist within.
3. Eco1C1G1T1.input.json : A JSON representation of the Library used by Cello, representing the input signals into the genetic circuit will exist within.
4. Eco1C1G1T1.output.json : A JSON representation of the Library used by Cello, representing the output of the genetic circuit.

Required Outputs. A text file that has the following fields
1. The Score : The new score of the circuit
2. *.UCF.json : The modified UCF file.
