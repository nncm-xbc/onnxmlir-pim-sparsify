# Compiler.py
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os
import sys
from MLP.mlp import *                  # Mini-Package for Multi Layer Perceptron

__location__       = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
args               = sys.argv
parameters_folder  = __location__ + "/" + args[1]


################################################################################################################################

class Program:
    # this class contains in a structured manner matrices and biases
    # which encode the semantics of the program network, in the same fashion
    # of the source code of a program
    def __init__(self, topology , activation_functions_list = None):
        if type(topology) == str: # passing file name
            params = load_network_params(topology)
            
            self.W = []
            self.b = []
            
            # "cast" the storage format 
            for p in params:
                self.W.append(p[0])
                self.b.append(p[1])
            
            self.activation_functions_list = [ 'RELU' for w in self.W[:-1] ] + ['LINEAR']
            self.topology = [ w.shape[1] for w in self.W]  + [self.W[-1].shape[0]]    

            print("Topology of the loaded network :" , self.topology)
        else:                     # empty nettwork of a given topology
            self.topology = topology
            self.activation_functions_list = activation_functions_list
            self.W = [ 0 for t in topology[:-1]]
            self.b = [ 0 for t in topology[:-1]]
        
        
        self.activation_functions = {}
        self.activation_functions['RELU']   = lambda x : jnp.maximum(0.,x)
        self.activation_functions['LINEAR'] = lambda x : x
        
    def random_init_weights(self):
        for layer in range(len(self.topology[:-1])):
            self.W[layer] = np.random.normal( size = (self.topology[layer + 1] , self.topology[layer] ) )**2
            size = self.W[layer].shape[0] * self.W[layer].shape[1] 
            for i in range(size - int(np.sqrt(size))):
                r = np.random.choice(len(self.W[layer]))
                c = np.random.choice(len(self.W[layer].T))
                self.W[layer][r,c] = 0.   # adding some sparsity
                
            self.b[layer] = np.random.normal( size = self.topology[layer + 1] )
    def print(self):
        for i,layer in enumerate(self.topology[:-1]):
            print("layer %d->%d "%(i,i+1))
            print("W.shape = %s \t b.shape = %s" % (str(self.W[i].shape),str(self.b[i].shape)))
    def visualize_weights(self):
        for w in self.W:
            print(w.shape)
            plt.figure(figsize=(10,10))
            plt.imshow(w > 0.)        
    
    def run(self, data):
        layer = data
        for f_name,weight,bias in zip(self.activation_functions_list,
                                      self.W,
                                      self.b):
            f     = self.activation_functions[f_name]
            layer = f(weight @ layer  + bias)
        return layer


    def random_test(self):
        return self.run(np.random.randn(self.topology[0]))
        
programma = Program(parameters_folder)
programma.random_init_weights()
print("Loading the neural network...")
print("Random test: " , programma.random_test() )

################################################################################################################################
#                                                   INTERMEDIATE REPRESENTATION                                                #
################################################################################################################################

#                                                >>>LIGHTWEIGHT TREE STRUCTURE<<<
#                                    * Simple class that stores data in a tree format 
#                                    * Projected to be performant bust most of all 
#                                    * Easy to manipulate

class c:
    # every element is a node in the tree.
    # the first argument denotes the name of the node ; the possible next arguments are a list of the sons.
    # Note that nodes without sons are simply leaf (eg. ADD, MUL,...)
    def __init__(self, ID, *args):
        self.id   = ID
        self.sons = list()
        for arg in args:
            self.sons.append(arg)
    def print(self,level = 0):
        print( ("\t" * level) + str(self.id) )
        for s in self.sons:
            s.print(level + 1)
    def __str__(self):
        ret = str(self.id)
        if len(self.sons) > 0:
            ret += "("
            for s in self.sons:
                ret += str(s)
                if s != self.sons[-1]:
                    ret += ','
            ret += ")"
        return ret
    def flatten(self):
        ret = []
        ret += [self.id]
        for s in self.sons:
            ret += s.flatten()
        return ret
            
#                                            >>>INTERMEDIATE REPRESENTATION SYNTHESIS<<<
#                                    * Given a MultiLayer Perceptron represented through the class
#                                    * Program produces a Tree based intermediate representation
#                                    * which is used by the compiler to generate assembly code.

def IR(program, compile_time_data = True):
    # for every layer
    ### print("y = input")
    ret = list()
    
    for layer,t in enumerate(program.topology[1:]):
        IR_instruction = c("COMMENT", c("START"))
        ### print(IR_instruction)
        ret.append(IR_instruction)
        
        f = program.activation_functions_list[layer]
        R_offset = program.topology[layer]
        for i in range(len(program.b[layer])): # for every row
            #print("R[%d] = 0" % (i))
            #print("MOVE( TEMP(%d) , CONST(0) )" % (i + R_offset))
            IR_instruction = c(
                    "MOVE",
                     c("TEMP" , c(i + R_offset) ),
                     c("CONST", c(0))
            )
            ### print(str(IR_instruction))
            ret.append(IR_instruction)
            
            
        for i in range(len(program.W[layer])): # for every row
            for j in range(len(program.W[layer].T)): # for every column
                if( program.W[layer][i,j] != 0):
                    # if there is zero is useless to compute the contribution
                    #print("->MOVE(TEMP(%d),BINOP(ADD,TEMP(%d),BINOP(MUL,CONST(%f),TEMP(%d)))) " % (i + R_offset,i + R_offset, program.W[layer][i,j],i )) # COMPILE DATA HYPOTHESIS
                    IR_instruction = c(
                        "MOVE",
                        c("TEMP",
                            c(i + R_offset)
                        ),
                        c("BINOP",
                            c("ADD"),
                            c("TEMP",
                                c(i + R_offset)
                            ),
                            c("BINOP",
                                c("MUL"),
                                c("CONST",
                                     c(program.W[layer][i,j])
                                 ),
                                c("TEMP",
                                 c(j)
                                 )
                            )
                        )
                    )
                    #print(IR_instruction)
                    ret.append(IR_instruction)
                else:
                    0
                    ### print("# here there was a 0 so we exploit sparsity ")
                    
        for i in range(len(program.b[layer])): # for every row
            # a priori in compile time since the amount of "repetitions" doens't scale quadratically , in opposite to weights
            #print("->MOVE(TEMP(%d),BINOP(ADD,TEMP(%d),CONST(%f)))" % (i, i + R_offset , program.b[layer][i]) )
            
            # For each output temporary i add the bias to it
            # Since after the add of the bias the only modification
            # That is performed on a temporary variable is the application of the 
            # activation function, i condensate the instructions
            # in order to avoid useless memory reads
            
            IR_instruction = c("MOVE",
                                c("CALL",
                                    c(f),
                                    c("TEMP",
                                     c(i + R_offset)
                                     ),
                                    c("BINOP",
                                     c("ADD"),
                                     c("TEMP",
                                       c(i + R_offset)
                                      ),
                                     c("CONST",
                                      c(program.b[layer][i])
                                      )
                                     )
                                 )
                                )
            ### print((IR_instruction))
            ret.append(IR_instruction)

        IR_instruction = c("COMMENT", c("END"))
        ### print(IR_instruction)
        ret.append(IR_instruction)
    return ret           
    

################################################################################################################################
#                                               REGISTER AND MEMORY ALLOCATION                                                 #
################################################################################################################################


#                                                >>>TEMPORARY VARIABLE STATS<<<
#                                    * A suitable class to contain statistical information
#                                    * regarding the usage of temporary variables. This is used
#                                    * for memory and register allocation
  
class TemporaryVariablesStatistics:
    def __init__(self):
        self.temp_usage_map = {}
    def increment(self,temp_variable):
        old_value = self.temp_usage_map.get(temp_variable)
        if old_value == None:
            old_value = 0
        self.temp_usage_map[temp_variable] = old_value + 1
    def get_data(self):
        return self.temp_usage_map
    def vectorize(self):
        arr = []
        for s in self.get_data():
            arr.append( [ s, self.get_data()[s]] )
        arr = np.array(arr)                                                              # builds a tempstable [ temp | usage ]
        arr = arr[ arr[:,1].argsort()[-1::-1] ]                                          # sort the tempstable by usage  (decreasing)
        return arr
    
    def print(self):
        for t in self.temp_usage_map:
            print("%d --> %d" % (t , self.temp_usage_map[t]) )


#                                                >>>REGISTER ALLOCATION CLASS<<<
#                                    * Class that provide tools for register allocation.
#                                    * It contains the mapping between the ids of temporary variables
#                                    * and register names.   
            
class RegisterAllocationData:
    def __init__(self):
        self.temp_reg_map = {}
    
    def get(self,temp_variable):
        return self.temp_reg_map[temp_variable]
    
    def insert(self,temp_variable, register):
        self.temp_reg_map[temp_variable] = register
        
    def get_data(self):
        return self.temp_reg_map
    
    def rename(self,old_reg_name, new_reg_name):                                         # Renames a register, inside the mapping from temporaries to regs.
        for t in self.temp_reg_map:                                                      # This means that if the mapping is defined as a collection 
            if self.temp_reg_map[t] == old_reg_name:                                     #                          TR := {T_i, R_i}_i
                self.temp_reg_map[t] = new_reg_name                                      # calling rename("R_old","R_new") a new collection where every instance of R_old is replaced
    
    def get_unitialized_temps(self):                                                     # All registers are initially unitialized. This means that temporary variables
        # returns the list of temps that have a register starting with "register_"       # are just known to be associated to register but it is impossible to know "a priori"
        ret = list()                                                                     # which register takes a variable. This is because the assignement depends on the
        for s in self.temp_reg_map:                                                      # assignement of the previous layer (the first layer is the only where I can decide
            if self.temp_reg_map[s].startswith("register_"):                             # on the spot which temporary variable assign to each register)
                ret.append(s)                                                            # The two methods here allow to interact with register, defining a dicotomy
        return ret                                                                       # Between unitialized and initialized registers
    
    def get_initialized_registers(self):
        # returns a list of the register ACTUALLY used (no place holder)
        ret = list()
        for s in self.temp_reg_map:
            if not self.temp_reg_map[s].startswith("register_"):
                ret.append(self.temp_reg_map[s])
        return ret
    
    def get_variables_list(self):                                                       # This method return the list of temporary variables that has been assigned to registers
        ret = list()
        for s in self.temp_reg_map:
            ret.append(s)
        return ret
    
    def get_input_temps(self, prev_layer_size):                                         # This method return the list of temporary variables that are responsible of 
        all_vars = np.array(self.get_variables_list())                                  # contain the output of the previous layer (in case  of the first layer this is the data)
        ret      = list()                                                               # The convention  used (which can be ignored by the user) is that the first variables
        for var in all_vars:                                                            # are the input ones, the others are related to output.
            if var < prev_layer_size:
                ret.append([var, self.temp_reg_map[var]])
        return np.array(ret)
    
    def get_output_temps(self, prev_layer_size):                                        # Same as above, here we return the temporary variables linked to input of the next layer
        all_vars = np.array(self.get_variables_list())                                  # (in case of the last layer this is the output of the network)
        ret      = list()
        for var in all_vars:
            if var >= prev_layer_size:
                ret.append([var, self.temp_reg_map[var]])
        return np.array(ret)
    
    def print(self):                                                                    # Debug function
        for t in self.temp_reg_map:
            print(t , "\t", self.temp_reg_map[t] )
            
            
    def contains(self,tmp_name):                                                        # Check if a temporary variable is allocated to registers
        return tmp_name in self.temp_reg_map
        
#                                                >>>MEMORY ALLOCATION LIST<<<
#                                    * Class that provide tools for memory allocation.
#                                    * It contains the mapping between the ids of temporary variables
#                                    * and register names.   

class MemoryAllocationData:
    def __init__(self):
        self.temp_mem_map = {}
        
    def insert(self,temp_variable, address):
        self.temp_mem_map[temp_variable] = address
        
    def batch_set(self, list_of_temps, list_of_addresses):                              # Given a set of temporary ids and a set of addresses
        for tmp_id, mem_addr in zip(list_of_temps,list_of_addresses):                   # It stores the mapping between them into the
            self.temp_mem_map[tmp_id] = mem_addr                                        # structure.
        
    def get_data(self):
        return self.temp_mem_map
    
    def get(self,x):
        return self.temp_mem_map[x]
    
    def get_variables_list(self):                                                       # Gets all the variables
        ret = list()                                                                    # stored into the structure
        for s in self.temp_mem_map:                                                     #
            ret.append(s)                                                               #
        return ret                                                                      #
    
    def get_input_temps(self, prev_layer_size):                                         # Get the variables stored into the
        all_vars = np.array(self.get_variables_list())                                  # structure, responsible for containing the output
        return all_vars[all_vars < prev_layer_size]                                     # of the previous layer
    
    def get_output_temps(self,prev_layer_size):                                         # Get the variables stored into the
        all_vars = np.array(self.get_variables_list())                                  # structure, responsible for containing the input
        return all_vars[all_vars >= prev_layer_size]                                    # of the next layer
    
    def print(self):                                                                    # Debug function
        for t in self.temp_mem_map:                                                     #
            print(t , "\t", self.temp_mem_map[t] )                                      #
            
################################################################################################################################
#                                          REGISTER AND MEMORY ALLOCATION STRATEGY                                             #
################################################################################################################################

#                                                >>>BLOCK SIGNALS<<<
#                                    * Block Signals are a structure able to represent, in each layer,
#                                    * the signal that represents the usage of temporary variables.
#                                    * Visually, can be imagined as a set of "clock" cycles that tick
#                                    * when each temporary variable is used  

class BlockSignals:
    def __init__(self, memory_allocation_object):
        # initialize an empty dictionary starting from the variables name
        self.memory_allocation_object = memory_allocation_object
        self.temp_signals_map = {}
        for t in memory_allocation_object.get_data():
            self.temp_signals_map[t] = []
        
    def add_tick(self, temp_variables):
        temp_variables          = np.intersect1d(temp_variables, self.memory_allocation_object.get_variables_list())
        all_temporary_variables = self.memory_allocation_object.get_variables_list()
        # push 0 in the lists of unused temps and 1 in the list of the used temp
        for t in all_temporary_variables:
            self.temp_signals_map[t].append(0)
        for t in temp_variables:
            self.temp_signals_map[t][-1] = 1.
            
    def get_data(self):
        return self.temp_signals_map
        
#                                                >>>FLOWS<<<
#                         * These set of classes represents flows of data, for which some
#                         * assembly code has to be generated.
#                         * For instance, if we have a Mem2Reg(addr,reg) it means that we have to
#                         *                 > load data from addr, store it in reg
       
class MemoryToRegisterFlow:
    # contains the information about the movement of information from a memory
    # cell to a register from a matrix mult to the next one
    def __init__(self, mem_address, register):
        self.mem_address = mem_address
        self.register    = register
    def print(self):
        print("M2R flow\t%s\t->\t%s" % (self.mem_address,self.register))
        
        
class RegisterToMemoryFlow:
    # contains the information about the movement of information from a register 
    # to a memory cell from a matrix mult to the next one
    def __init__(self, register, mem_address):
        self.mem_address = mem_address
        self.register    = register
    def print(self):
        print("R2M flow\t%s\t->\t%s" % (self.register,self.mem_address))

class RegisterRenameFlow:
    # contains the information about the movement of information from a register 
    # to a memory cell from a matrix mult to the next one
    def __init__(self, register_placeholder, register_name):
        self.register_placeholder = register_placeholder
        self.register_name        = register_name
    def print(self):
        print("R2R flow\t%s\t->\t%s" % (self.register_placeholder,self.register_name))


#                                            >>>INTERFACE COMMUNICATION<<<
#                         * Allow to define,  monolithically, flows from a layer to the next
#                         * The code generator can access an object of this class to generate
#                         * code relative to flows.
        
class InterfaceCommunication:
    # contains the list of movements "flows" between two matrix mult blocks
    def __init__(self):
        self.reg2memFlows = list()
        self.mem2regFlows = list()
        self.reg2regFlows = list()
    def insert(self,flow):
        if type(flow).__name__ == "RegisterToMemoryFlow":
            self.reg2memFlows.append(flow)
        else:
            if type(flow).__name__ == "MemoryToRegisterFlow":
                self.mem2regFlows.append(flow)
            else:
                if type(flow).__name__ == "RegisterRenameFlow":
                    self.reg2regFlows.append(flow)
                    
    def getReg2MemFlows(self):
        return self.reg2memFlows
    def getMem2RegFlows(self):
        return self.mem2regFlows
    def getReg2RegFlows(self):
        return self.reg2regFlows
    
    def print(self):
        reg2mem = self.getMem2RegFlows()
        mem2reg = self.getReg2MemFlows()
        reg2reg = self.getReg2RegFlows()
        for rm in reg2mem:
            rm.print()
        for mr in mem2reg:
            mr.print()
        for rr in reg2reg:
            rr.print()

################################################################################################################################
#                                                        ALLOCATOR CLASS                                                       #
################################################################################################################################