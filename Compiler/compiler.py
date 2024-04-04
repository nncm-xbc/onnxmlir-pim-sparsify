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

#                       * This very big class handles all the tools that we constructed so far.
#                       * 
#                       *  (function calls structture)
#                       *
#                       *  init(IR) 
#                       *       |
#                       *       +--> most_used_temps(IR) => stats                              // produces the statistics regarding the frequency of usage of each temporary 
#                       *                       |
#                       *                       |   
#                       *            register_allocation_and_memory_alloc_init(stats)          // takes the statistics and uses them to initialize the reg allocator and mem allocator
#                       *                       |                                              // which means, insert the variables into the allocators without expliciting declaring where to allocate
#                       *                       |                                              // them in memory. (Registers in the first layer are allocated here)
#                       *                       |
#                       *           compute_signals(IR) => signals                             // Computes the "ticks" of the frequency signals, see BlockSignals
#                       *                       |
#                       *           compute_signals_distance_matrix(signals) => D              // Computes a NxN matrix where the ij-th element represent the "uncorrelation" between temp_i and temp_j 
#                       *                       |
#                       *           memory_allocation(D)                                       // Runs the memory allocation procedure according to the information extracted by DATA
#                       *                       |
#                       *                       +------->

from sklearn.manifold import MDS

class Allocator:
    def __init__(self, ir, program, register_names):
        self.register_allocation_data = []  # register allocations for every matrix multiplication operation
        self.memory_allocation_data   = []  # memory allocations for every matrix multiplication operation
        self.interfaces               = {}
        self.register_names           = register_names
        self.program                  = program
        # ------------------------------
        # Register allocation subroutine
        
        # compute the usage of each temp
        temp_statistics = self.most_used_temps(ir)   # produces a list of register statistics objects
        
        # 
        self.register_allocation_and_memory_alloc_init(temp_statistics, register_names)
        self.memory_allocation(
                self.compute_signals_distance_matrix(
                    self.compute_signals(ir)
                )
        )
        
    ########################################################################################
    # Input and output masks
    ########################################################################################
    
    def get_input_mapping(self):
        ret = {}
        def is_in_first_layer_input(temp):
            if temp < self.program.topology[0]:
                return True
            return False
        
        for temp in self.register_allocation_data[0].get_data():
            if is_in_first_layer_input(temp):
                ret[temp] = ("reg", self.register_allocation_data[0].get_data()[temp]) 
        
        for temp in self.memory_allocation_data[0].get_data():
            if is_in_first_layer_input(temp):
                ret[temp] = ("mem", 4 * int(self.memory_allocation_data[0].get_data()[temp]))
        return ret
    
    def get_output_mapping(self):
        ret = {}
        def is_in_last_layer_output(temp):
            if temp >= self.program.topology[-2]:
                return True
            return False
        
        for temp in self.register_allocation_data[-1].get_data():
            if is_in_last_layer_output(temp):
                ret[temp - self.program.topology[-2]] = ("reg", self.register_allocation_data[-1].get_data()[temp]) 
        
        for temp in self.memory_allocation_data[-1].get_data():
            if is_in_last_layer_output(temp):
                ret[temp - self.program.topology[-2]] = ("mem", 4 * int(self.memory_allocation_data[-1].get_data()[temp]))
        return ret
    ########################################################################################
    # Register allocation
    ########################################################################################
    
    def most_used_temps(self, ir):
        # IN   : takes as input an intermediate representation
        # OUT  : produces a list of TemporaryVariableStatistics objects, one for each matrix mult
        statistics_per_block = list()
        
        for ir_instruction in ir:                                                            # iterate over the IR statements
            if(ir_instruction.id == "COMMENT"):                                              # 
                if(ir_instruction.sons[0].id == "START"):
                    statistics_per_block.append(TemporaryVariablesStatistics())              # i create a temporaryvariablestatistcs
            else:
                unrolled_ir = ir_instruction.flatten()                                       # unroll the statemenet
                temps_in_statement   = list()                                                # container for temps variables in the current statement
                for u,val in zip(unrolled_ir[:-1],unrolled_ir[1:]):  
                    if u == "TEMP": 
                        temps_in_statement.append(val)                                       
                                                                                             # now "temps_in_statement" contains only the values of the temporary variables
                for t in temps_in_statement:                                                 # count the usage of each temporal 
                    statistics_per_block[-1].increment(t)                                    # the current temporary variable statistics is updated 
        return statistics_per_block
    
    def register_allocation_and_memory_alloc_init(self, statistics_list , register_names):
        # IN  : a statistics list obtained from  most_used_temps , register names
        # OUT : a registerAllocation object
        
        # convert the dictionary to an array
        temp_stats_per_block = list()
        first_block = True
        for stat in statistics_list:
            # I transform the dictitonary in a sorted-by-usage vector
            arr = stat.vectorize()
            #print(arr)
            
            # I initialize a Registerallocation  object
            reg_data = RegisterAllocationData()   
            
            # for every register i take an element, starting from the beginning, of the array
            temp_var_count = 0
            for r_id,r in enumerate(register_names):
                if temp_var_count >= len(arr):
                    break
                if first_block:     # registers are decided a priori only in the first block
                    reg_data.insert(arr[temp_var_count,0],r)                                  # i add as a used register the temporary variables with more usage
                else:
                    reg_data.insert(arr[temp_var_count,0],"register_%d" % r_id)
                temp_var_count += 1
            self.register_allocation_data.append(reg_data)                                    # i append the register allocation data obtained to the list of RAD
            
            # I also initialize the "slots" for the memory allocation data
            mem_data = MemoryAllocationData()                                                 
            for t in range(temp_var_count, len(arr)):
                mem_data.insert(arr[t,0], -1)                                                 # initializa with -1
            self.memory_allocation_data.append(mem_data)                                      # i add them to the list
            first_block = False
        
        ###for r in self.register_allocation_data:
        ###    print("+---------+")
        ###    r.print()
        
        return 0
    
    ########################################################################################
    # Memory allocation
    ########################################################################################

    def compute_signals(self, ir):
        # takes as input an intermediate repr and a register allocation output
        signals_per_block = list()
 
        curr_alloc_block = 0
    
        for ir_instruction in ir:
            if(ir_instruction.id == "COMMENT"):
                if(ir_instruction.sons[0].id == "START"):
                    curr_memory_alloc_block = self.memory_allocation_data[curr_alloc_block]
                    signals_per_block.append(BlockSignals(curr_memory_alloc_block))                       
                    curr_alloc_block += 1

            unrolled_ir = ir_instruction.flatten()
            temps_in_statement   = list()
            for u,val in zip(unrolled_ir[:-1],unrolled_ir[1:]):
                if u == "TEMP":
                    temps_in_statement.append(val)      
                    
            signals_per_block[-1].add_tick(temps_in_statement)
        return signals_per_block
    
    def compute_signals_distance_matrix(self, signals):
        # IN   : takes as input a collection of TempVarSignals
        # OUT  : produces a 
        inverse_mappings = list()
        mappings = list()          # mapping between the rows of the matrix and the temp_var
        Ds = list()                # list of matrices
        
        for signals_block in signals:
            D = np.zeros((len(signals_block.get_data()),len(signals_block.get_data())))
            mapping = {}
            inverse_mapping = {}
            for i,a in enumerate(signals_block.get_data()):
                mapping[i] = a
                inverse_mapping[a] = i
                for j,b in enumerate(signals_block.get_data()):  
                    v_a = np.arange(len(signals_block.get_data()[a]))[np.array(signals_block.get_data()[a]) == 1.]
                    v_b = np.arange(len(signals_block.get_data()[b]))[np.array(signals_block.get_data()[b]) == 1.]
                    distanza = 0.5 * (np.mean([ np.min(np.abs(s_1 - v_b)) for s_1 in v_a]) + np.mean([ np.min(np.abs(s_2 - v_a)) for s_2 in v_b]))
                    D[i,j] = distanza
            Ds.append(D)
            mappings.append(mapping)
            inverse_mappings.append(inverse_mapping)
        return Ds, mappings, inverse_mappings
    
    def anneal(self,
               unconstrainedTemps,     # temporary unconstrained
               addresses,              # list of available memory addresses
               Ds,                     # distance matrix
               inverse_mapping):       # mapping between rows of the distance matrix and temps
        ####print(inverse_mapping)
        # initialize a random association
        
        association = []
        for u,a in zip(unconstrainedTemps,addresses):
            association.append([u,a])
        association = np.array(association)
        
        # nothing to optimize actually...
        if(len(unconstrainedTemps) == 1):
            return association
        
        decay = 1e-3 #decay = 1e-3
        T = 10.
        T_end = 1e-3
        
        inverse_mapping_unconstrained = {}
        for i,u in enumerate(unconstrainedTemps):
            inverse_mapping_unconstrained[u] = i
        
        Ds_unconstrained = Ds[
            [ inverse_mapping[u] for u in unconstrainedTemps], :
        ][ :, [ inverse_mapping[u] for u in unconstrainedTemps]]
        
        
        def build_D_reconstructed(association):
            D_reconstructed = np.zeros(Ds_unconstrained.shape)
            for a in association:
                for b in association:
                    D_reconstructed[inverse_mapping_unconstrained[a[0]], inverse_mapping_unconstrained[b[0]]] = np.abs( a[1] - b[1] )
            return D_reconstructed
        
        def std(M):
            return (M - M.min(axis = 1)[:,None]) / (M.max(axis = 1) - M.min(axis = 1))[:,None]
        
        def cost(association):
            D_reconstructed = build_D_reconstructed(association)
            return np.linalg.norm(
                std(D_reconstructed) - std(Ds_unconstrained)
            )
        # ANNEALING
        
        anneal_iterations = (np.log(T_end) - np.log(T)) / np.log(1 - decay)
        anneal_count = 0
        while T > T_end:
            T = (1. - decay) * T

            mossa            = np.arange(len(association)).astype(int)
            # swap two random
            a = np.random.choice(len(mossa))
            b = np.random.choice(len(mossa))
            tmp = mossa[a]
            mossa[a] = mossa[b]
            mossa[b] = tmp
            new_association      = association.copy()
            new_association[:,1] = association[:,1][mossa]
            
            dE = cost(new_association) - cost(association)
            
            if dE <= 0.:
                association = new_association
            else:
                if( np.random.uniform() > np.exp( - T)):
                    association = new_association
            
            if anneal_count % int( anneal_iterations / 10 ) == 0:
                print("cost = %.2f" % cost(association))
                
            anneal_count += 1
        
        return association
    
    
    
    def density_optimizer_memory_subset_for_output(self,
                                                   memory,
                                                   constrained, 
                                                   m_i,
                                                   unconstrained_temps_size):
        # i get the already allocated addresses
        m_i_on_constraint = [ 
                m_i.get(c)
            for c in constrained
        ]
        ###print("|unconstrained| = %d" % unconstrained_temps_size)
        ###print("constrained addresses = ", m_i_on_constraint)
        memory_mask = np.arange(len(memory))[ [ not(x in m_i_on_constraint ) for x in np.arange(len(memory))] ]
        ###print("MEMORY MASK =" , memory_mask)
        
        rows = list()
        densities = list()
        
        for j in range( len(memory) - len(m_i_on_constraint) - unconstrained_temps_size + 1):
            row = np.zeros(len(memory))
            
            for mem_add_const in m_i_on_constraint:
                row[mem_add_const] = -1.
            
            row[memory_mask[j:j+unconstrained_temps_size]] = 1.
            
            density = lambda r : (r != 0 )[ (r != 0).argmax() : (len(r) - (r != 0)[-1::-1].argmax())].mean()
            rows.append(row)
            densities.append(density(row))
            
        if(len(rows) == 1):
            return memory[rows[0] > 0.]
        else:
            densities = np.array(densities)
            rows      = np.array(rows)
            #plt.figure()
            #plt.imshow(rows)
            # choose the best
            # among the maximal densities
            ###print("@@@",rows)
            ###print("###",densities)
            ###print("***",densities == densities.max())
            available_configurations = rows[densities == densities.max()]
            # choose the one where the first unconstrained address is closer to the last constrained address
            ###print(available_configurations)
            configuration = available_configurations[0]
            for a in available_configurations:
                first_one     = configuration.argmax()
                last_minusone = len(configuration) - configuration[-1::-1].argmax() 
                distance_conf = np.abs(last_minusone - first_one)

                afirst_one     = a.argmax()
                alast_minusone = len(a) - a[-1::-1].argmax()
                adistance_conf = np.abs(alast_minusone - afirst_one)
                
                if adistance_conf < distance_conf:
                    configuration = a


            ####print("CHOSEN = ", configuration)
            ####print( m_i_on_constraint )
            ####print( memory[configuration > 0.] )
            return memory[configuration > 0.]
        

####################################################################################################################
#####                  #############################################################################################
#####   huge function  #############################################################################################
#####                  #############################################################################################
####################################################################################################################

    def memory_allocation(self, DS_MAPPINGS):
        
                   

        
        # i define a fake memory just for debug
        memory = np.arange(
            np.max(
                [
                       np.max(len(m.get_variables_list()))
                    for m in self.memory_allocation_data 
                ]
            )
        )
        
        ####print("MEMORY = ", memory)
        ####print("\nMEM ALLOC DATA [len DS = %d]\n" % len(DS_MAPPINGS[0]))
        
        # compute m1
        # every memory temp is an unconstrained temp
        print("\tMatrix Multiplication %d" % 0)
        T_mem_1       = self.memory_allocation_data[0].get_variables_list()
        mem_addresses = np.arange(0, len(T_mem_1))
        Ds_1                      = DS_MAPPINGS[0][0]
        mapping_1                 = DS_MAPPINGS[1][0]
        inverse_mapping_1         = DS_MAPPINGS[2][0]
        # optimize the                 
        if len(T_mem_1) > 0:
            print("Optimizing memory allocation of unconstrained temps...")
            temp_addr_mapping = self.anneal( T_mem_1, 
                                        mem_addresses,
                                        Ds_1,
                                        inverse_mapping_1
                                      )
            for mappa in temp_addr_mapping:
                self.memory_allocation_data[0].insert(
                    mappa[0],
                    mappa[1]
                )
        
        T                       =  np.arange( self.program.topology[0] + self.program.topology[1] )
        T_in_curr, T_out_curr   = T_in_prev , T_out_prev  =  set(T[T < self.program.topology[0]]) , set(T[T >= self.program.topology[0]])
        T_mem_curr, T_reg_curr  = T_mem_prev , T_reg_prev =  set(self.memory_allocation_data[0].get_variables_list()), set(self.register_allocation_data[0].get_variables_list())
        

        for i in range(1, len(DS_MAPPINGS[0]) ): 
            print("\n\tMatrix Multiplication %d" % i)
            ##############
            
            T                       = np.arange( self.program.topology[i] + self.program.topology[i + 1] )
            # Define the IO partition
            T_in_curr , T_out_curr  = set(T[T < self.program.topology[i]]) , set(T[T >= self.program.topology[i]])
            # Define the MR partition
            T_mem_curr , T_reg_curr = set(self.memory_allocation_data[i].get_variables_list()), set(self.register_allocation_data[i].get_variables_list())    
        
            ### Define the mappings when the input is batch
            
            # define the mapping from curr input to previous layer output
            phi_inv      = lambda t_in_curr  : set((len(T_in_prev) + np.array(list(t_in_curr))))
            # define the mapping from previous layer output to curr input
            phi          = lambda t_out_prev : set((np.array(list(t_out_prev)) - len(T_in_prev)))
            
            ### Redefine the mapping when the input is single
            # define the mapping from curr input to previous layer output
            single_phi_inv      = lambda t_in_curr  : len(T_in_prev) + t_in_curr
            # define the mapping from previous layer output to curr input
            single_phi          = lambda t_out_prev : t_out_prev - len(T_in_prev)
            
            
            # define the flows
            self.interfaces[i-1,i] = InterfaceCommunication()
           
            # FreeRegisters
            freeRegisters = set()
            for r in self.register_names:
                freeRegisters.add(r)
            ####print(freeRegisters)
            
            
            # if some output stays in registers when it becomes input we dont want to move it
            for t in T_in_curr.intersection(T_reg_curr):
                if single_phi_inv(t) in ( T_out_prev.intersection(T_reg_prev)):
                    r  = self.register_allocation_data[i - 1].get(
                                                               single_phi_inv(t)
                                                           )
                    self.register_allocation_data[i].insert(t, 
                                                            r
                                                           )
                    freeRegisters.remove(r)
                    
            # from memory to registers
            for t in T_in_curr.intersection(T_reg_curr):
                if single_phi_inv(t) in T_out_prev.intersection(T_mem_prev):
                    r = list(freeRegisters)[0]
                    self.register_allocation_data[i].insert(t, r)
                    freeRegisters.remove(r)
                    self.interfaces[i-1,i].insert(
                        MemoryToRegisterFlow(
                            self.memory_allocation_data[i - 1].get(single_phi_inv(t)),
                            r
                        )
                    )
                    
            # outputs that go into registers
            for t in T_out_curr.intersection(T_reg_curr):
                r = list(freeRegisters)[0]
                self.register_allocation_data[i].insert(t, r)
                freeRegisters.remove(r)
            
            
            # Memory
            
            UnconstrainedTemps = T_mem_curr.copy()
            ConstrainedTemps   = set()
            
            for t in T_in_curr.intersection(T_mem_curr):
                if single_phi_inv(t) in T_out_prev.intersection(T_mem_prev):
                    self.memory_allocation_data[i].insert( 
                        t , 
                        self.memory_allocation_data[i - 1].get(single_phi_inv(t))
                    )       
                    UnconstrainedTemps.remove(t)
                    ConstrainedTemps.add(t)
                    
            if len(UnconstrainedTemps) > 0:
                print("Optimizing memory allocation of unconstrained temps...")
                
                MemoryAddressImageOfMi = self.density_optimizer_memory_subset_for_output(
                    memory,                           # memory object
                    ConstrainedTemps,                 # set temporaries that are constrained
                    self.memory_allocation_data[i],   # funzione m_i
                    len(UnconstrainedTemps)           # length of the unconstrained temps to place  
                ) # returns a list of addresses that suit the unconstrained variables

                
                Ds              = DS_MAPPINGS[0][i]
                mapping         = DS_MAPPINGS[1][i]
                inverse_mapping = DS_MAPPINGS[2][i]
                
            
                temp_addr_mapping = self.anneal(    UnconstrainedTemps, 
                                                    MemoryAddressImageOfMi,
                                                    Ds,
                                                    inverse_mapping
                                                  )
                for a in temp_addr_mapping:
                    self.memory_allocation_data[i].insert(a[0],a[1])
            
            
            for t in T_in_curr.intersection(T_mem_curr):
                if single_phi_inv(t) in T_out_prev.intersection(T_reg_prev):
                    self.interfaces[i-1,i].insert(
                        RegisterToMemoryFlow(
                            self.register_allocation_data[i-1].get(
                                single_phi_inv(t)
                            ),
                            self.memory_allocation_data[i].get(t)
                        )
                    )
            
            
            # save the previous partitions
            T_in_prev , T_out_prev   = T_in_curr  , T_out_curr  
            T_mem_prev , T_reg_prev  = T_mem_curr , T_reg_curr  
            
################################################################################################################################
#                                                      INTERFACE MANAGER                                                       #
################################################################################################################################     
#                       * As we said we can have Mem2Reg flows and Reg2Mem flows.
#                       * This translates in something maybe like this
#                                       $0 --> 0x1234               // A
#                                       [0x1234] --> $1             // B
#                                       $1 --> 0x1235               // C
#                                       [0x1235] --> $0             // D
#                       * Which mean that we have a cyclic dependece in flows. In fact, the dependecy graph should look like this
#
#                                   $0 --> 0x1234 --> $1 --> 0x1235---+
#                                   ^                                 |
#                                   '---------------------------------+
#                       
#                       * We notice that if we copy $0$ into $X(=$0) we can rewrite the graph as
#                                   $0         $X-> 0x1234 --> $1 --> 0x1235---+
#                                   ^                                          |
#                                   '------------------------------------------+
#                         Removing the ciclicity and allowing us to write, starting from $0 and going backwoard
#                       * Before generating code, we have to take care of this situation, so the current algorithm manages 
#                       * This kind of ciclyc dependency in the described way, generating the code that handles all the flows 
#                       * between layers
      
import networkx as nx
def interfaces_manager(interface,buffer_register_1):
    global ret_
    ret_   = ""
    def codeprint_function(stringa):
        global ret_
        ret_ += stringa + "\n"
    
    
    G = nx.DiGraph()
    selectable_registers = set()
    for r2m in interface.getReg2MemFlows():
        G.add_edge("reg_" + str(r2m.register), "mem_" + str(r2m.mem_address))
        G.nodes["mem_" + str(r2m.mem_address)]['in'] = "reg_" + str(r2m.register)
        G.nodes["reg_" + str(r2m.register)  ]['out'] = "mem_" + str(r2m.mem_address) 
        
        selectable_registers.add("reg_" + str(r2m.register))
        
    for m2r in interface.getMem2RegFlows():
        G.add_edge("mem_" + str(m2r.mem_address),"reg_" + str(m2r.register))
        
        G.nodes["mem_" + str(m2r.mem_address)]['out'] = "reg_" + str(m2r.register)
        G.nodes["reg_" + str(m2r.register)  ]['in'] = "mem_" + str(m2r.mem_address) 
        
        selectable_registers.add("reg_" + str(m2r.register))


    # until no more selectable registers
    while len(selectable_registers) > 0:
    #   head <- select a random SELECTABLE register (each thread or loop contains AT LEAST ONE register by the way flows are defined)
        head_id = list(selectable_registers)[0]
        selectable_registers.remove(head_id)
        head    = G.nodes[head_id]
        
    #   cursor <- head
        cursor_id = head_id
        cursor    = head
        
    #   while cursor != head AND cursor != NIL
        
        if cursor.get("out") != None:
            cursor_id = cursor.get("out")
            cursor = G.nodes[cursor_id]
            
            while cursor != head:
        #       cursor <- cursor.next
                if cursor.get("out") != None:
                    cursor_id = cursor.get("out")
                    cursor    = G.nodes[cursor_id]
                else:
                    break
    # 
    #   if cursor.next = NIL
        if cursor.get('out') == None:
    #       // this is a thread
    #       while cursor.prev.prev != NIL
            while G.nodes.get(cursor.get("in"))  != None:
                cursor_father_id = cursor.get("in")
                
                if cursor_id in selectable_registers:
                        selectable_registers.remove(cursor_id)
                if cursor_father_id in selectable_registers:
                        selectable_registers.remove(cursor_father_id)
                        
    #           codeprint [ move data from cursor.prev to cursor ]
                    
                # if source is register
                if cursor_father_id.startswith("reg_"):
                    register = cursor_father_id.split("_")[1]
                    address  = 4 * int(cursor_id.split("_")[1])
                    # // dest is memory
                    codeprint_function("FSTS %s,[r7,#%s]" %  (register, address))
                # if source is memory
                else:
                    # // dest is register
                    register  = cursor_id.split("_")[1]
                    address = 4 * int(cursor_father_id.split("_")[1])
                    codeprint_function("FLDS %s,[r7,#%s]" %  (register, address))
    #           cursor = cursor.prev
                cursor_id = cursor.get("in")
                cursor    = G.nodes[cursor_id]
        else:
    #       // this is a loop
    #       codeprint [ copy head into BUFFER_REGISTER_1] // head_copy
            ### codeprint_function("# move from %s to %s" % (head_id , "BUFFER_REGISTER_1"))
            register = head_id.split("_")[1]
            codeprint_function("VMOV.F32 %s,%s" % (buffer_register_1 , register))
    #       cursor <- head // asserted by the if
    #       while cursor.prev != head
            while cursor.get("in") != head_id:
    #           if cursors is a register
    #                cursor.selectable = False
                codeprint_function("#printing %s" % cursor_id)
                cursor_father_id = cursor.get("in")
                
                if cursor_id in selectable_registers:
                        selectable_registers.remove(cursor_id)
                if cursor_father_id in selectable_registers:
                        selectable_registers.remove(cursor_father_id)
    #           codeprint [ move data from cursor.prev to cursor ]    
                # if source is register
                if cursor_father_id.startswith("reg_"):
                    register = cursor_father_id.split("_")[1]
                    address  = 4 * int(cursor_id.split("_")[1])
                    # // dest is memory
                    codeprint_function("FSTS %s,[r7,#%s]" %  (register, address))
                # if source is memory
                else:
                    # // dest is register
                    register  = cursor_id.split("_")[1]
                    address = 4 * int(cursor_father_id.split("_")[1])
                    codeprint_function("FLDS %s,[r7,#%s]" %  (register, address))
    #           cursor = cursor.prev
                cursor_id = cursor.get("in")
                cursor    = G.nodes[cursor_id]
    #       codeprint [ copy BUFFER_REGISTER_1 into cursor] // head_copy
            ###codeprint_function("# move from %s to %s" % ("BUFFER_REGISTER_1", cursor_id))
            register = cursor_id.split("_")[1]
            codeprint_function("VMOV.F32 %s,%s" % (register, buffer_register_1))    
    return ret_[:-1]
 

            
################################################################################################################################
#                                                      COMPILER FUNCTION                                                       #
################################################################################################################################
                   
 
import struct
def compiler(rete, registers, sparsify = False, r7offset = 0):
    

    global ret
    ret   = ""
    def codeprint(stringa):
        global ret
        ret += stringa + "\n"
    
    reg_0 = registers[0]
    reg_1 = registers[1]
    
    # debug, we don't actually read a file but we generate it randomly
    #rete = Program([10,30,30,2], ["RELU","RELU","LINEAR"])
    #rete.random_init_weights()
    intermediate_representation = IR(rete)
    for i in intermediate_representation:
        print(i.flatten())
    
    allocator   = Allocator(intermediate_representation , rete , registers[3:])
    input_mask  = allocator.get_input_mapping()    # dictionary that maps inputs entries to registers/memory units
    output_mask = allocator.get_output_mapping()   # dictionary that maps outputs entries to register/memory units
    
    
    
    
    asm_code = []
    
    zero_register     = registers[0]
    buffer_register_1 = registers[1]
    buffer_register_2 = registers[2]
    codeprint("MOV r7, %d \t ; # address offset" % r7offset)
    codeprint("VSUB.F32 %s,%s,%s \t ; # initalize the zero register"
         % (
             zero_register,
             zero_register,
             zero_register
         )
         )
    
    #for m in allocator.memory_allocation_data:
    #    m.print()
    #    print("----")
    
    cursor_allocator = 0
    label_counter = 0
    
    for ir_statement in intermediate_representation:
        flattened = ir_statement.flatten()
        #print(flattened)
        if(flattened[0] == "COMMENT"):
            if(flattened[1] == "END"):
                #print("\n # NUOVO BLOCCO \n")
                codeprint("######################################## HERE GOES THE INTERFACE")
                if (cursor_allocator) < len(allocator.interfaces):
                    codeprint("# LEGGO L'INTERFACCIA [%d,%d]" % (cursor_allocator, cursor_allocator + 1))
                    interfaccia = allocator.interfaces[cursor_allocator, cursor_allocator + 1]

                    codeprint( interfaces_manager(interfaccia,buffer_register_1) )

                    #interfaccia.print()
                cursor_allocator += 1
        
        if(flattened[0] == "MOVE" 
           and 
           flattened[3] == "CONST"
          ): # set the temporary variable
            tmp_name =  flattened[2]
            val       = flattened[4]
            # we have to understand if tmp_name is a register or not
            if(allocator.register_allocation_data[cursor_allocator].contains(tmp_name)):
                # variable is in registers
                register = allocator.register_allocation_data[cursor_allocator].get_data()[tmp_name]
                #print("MOV TO $%s THE VALUE #%f" % (
                #      register,
                #      val)
                #     )
                codeprint("VMOV.F32 %s, %s\t ; # set the register to 0 " % (register,zero_register))
            else:
                # variable is in memory
                address = 4 * allocator.memory_allocation_data[cursor_allocator].get_data()[tmp_name]
                #print("STORE THE VALUE 0 IN ADDRESS %d" % address)
                codeprint("FSTS %s, [r7,#%d]" % (zero_register,address) )
                
                
        if(flattened[0] == "MOVE"
           and
           flattened[3] == "BINOP"
           and 
           len(flattened ) == 13
          ): # addition and multiply
            neuron_dest   = flattened[2]
            neuron_source = flattened[12]
            weight        = flattened[10]
            
            
            if(allocator.register_allocation_data[cursor_allocator].contains(neuron_source)):
                register = allocator.register_allocation_data[cursor_allocator].get_data()[neuron_source]
                weight_bytes = struct.pack('f', weight)
                weight_bytes_upper = struct.unpack('H',weight_bytes[:2])
                weight_bytes_lower = struct.unpack('H',weight_bytes[2:])
                
                codeprint("MOVW r1,#%d \t ; # move the most significative bits in r3" % weight_bytes_upper[0])
                codeprint("MOVT r1,#%d \t ; # move the least significative bits in r3" % weight_bytes_lower[0])
                codeprint("VMOV.F32 %s,r1 \t ; # copy r3 in buffer_register_1" % (buffer_register_1))
                codeprint("VMUL.F32 %s,%s,%s \t ; # save in buffer_register_2 the multiplication " % (
                        buffer_register_2,
                        buffer_register_1,
                        register
                ))
                #print("MULT $%s BY #%f AND SAVE IT IN $%s" % (
                #      allocator.register_allocation_data[cursor_allocator].get_data()[neuron_source],
                #      weight,
                #      reg_0)
                #     )
            else:
                # variable is in memory
                address = 4 * allocator.memory_allocation_data[cursor_allocator].get_data()[neuron_source]
                weight_bytes = struct.pack('f', weight)
                weight_bytes_upper = struct.unpack('H',weight_bytes[:2])
                weight_bytes_lower = struct.unpack('H',weight_bytes[2:])
                
                codeprint("FLDS %s,[r7, #%d] \t ; # copy the value in the address on the buffer_register_2" % (buffer_register_2, address))
                codeprint("MOVW r1,#%d \t ; # move the most significative bits in r3" % weight_bytes_upper[0])
                codeprint("MOVT r1,#%d \t ; # move the least significative bits in r3" % weight_bytes_lower[0])
                codeprint("VMOV.F32 %s,r1 \t ; # copy r3 in buffer_register_1" % (buffer_register_1))
                codeprint("VMUL.F32 %s,%s,%s \t ; # save in buffer_register_2 the multiplication " % (
                        buffer_register_2,
                        buffer_register_1,
                        buffer_register_2
                ))    
                #print("LOAD IN REGISTER $%s THE ADDRESS %d"  % (
                #            reg_0,
                #            allocator.memory_allocation_data[cursor_allocator].get_data()[neuron_source])
                #     )
                #print("MULT $%s BY #%f AND SAVE IT IN $%s" % (
                #      reg_0,
                #      weight,
                #      reg_0
                #      )
                 #    )
                    
            # NOW BUFFER_REGISTER_2 CONTAINS weight * input
        
            if(allocator.register_allocation_data[cursor_allocator].contains(neuron_dest)):
                #print("ADD $%s to $%s" % (
                #      reg_0,
                #      allocator.register_allocation_data[cursor_allocator].get_data()[neuron_dest])
                #     )
                
                # dest += weight * input
                codeprint("VADD.F32 %s,%s,%s \t ; # " %( 
                    allocator.register_allocation_data[cursor_allocator].get_data()[neuron_dest],
                    buffer_register_2,
                    allocator.register_allocation_data[cursor_allocator].get_data()[neuron_dest]))
                
            else:
                # variable is in memory
                address = 4 * allocator.memory_allocation_data[cursor_allocator].get_data()[neuron_dest]
                #print("LOAD IN REGISTER $ %s THE ADDRESS %d" % (reg_1,address) )               
                codeprint("FLDS %s,[r7, #%d] \t ; # load the content of the output neuron in buffer_register_1 " % (buffer_register_1, address))
                codeprint("VADD.F32 %s,%s,%s \t ; # buffer_register_1 += weight * input"  % (buffer_register_1, buffer_register_2, buffer_register_1))
                codeprint("FSTS %s,[r7, #%d] " % (buffer_register_1, address))
                
        if(flattened[0] == "MOVE"
           and
           flattened[1] == "CALL"
          ): # adding the bias and applying the activation function
            
            tmp_name = flattened[4]  # name of the temporary variable
            f_name   = flattened[2]  # name of the activation function
            bias     = flattened[-1] # bias
            
            if(allocator.register_allocation_data[cursor_allocator].contains(tmp_name)):
                # the variable is in a register
                register = allocator.register_allocation_data[cursor_allocator].get_data()[tmp_name]
                #print("ADD TO $%s VALUE #%f " %
                #             (
                #                 register,
                #                 bias
                #             )
                #         )              
                
                register = allocator.register_allocation_data[cursor_allocator].get_data()[tmp_name]
                bias_bytes = struct.pack('f', bias)
                bias_bytes_upper = struct.unpack('H',bias_bytes[:2])
                bias_bytes_lower = struct.unpack('H',bias_bytes[2:])
                
                codeprint("MOVW r1,#%d \t ; # move the most significative bits in r3" % bias_bytes_upper[0])
                codeprint("MOVT r1,#%d \t ; # move the least significative bits in r3" % bias_bytes_lower[0])
                codeprint("VMOV.F32 %s,r1 \t ; # copy r3 in buffer_register_1" % (buffer_register_1))
                codeprint("VADD.F32 %s,%s,%s \t ; # add the bias" % (
                        register,
                        buffer_register_1,
                        register
                ))
                
                
                if f_name == 'RELU':
                    #the function is RELU
                    #print("GTZ $%s TO $%s" %   # if greater than 0 save 1 in $0 else 0 in $0
                    #         (
                    #             register,
                    #             reg_0
                    #         )
                    #     )
                    #print("MULT $%s, $%s, $%s" %
                    #         (
                    #             register,
                    #             reg_0,
                    #             register
                    #         )
                    #     )
                    codeprint("# RELU")
                    codeprint("vcmpe.f32  %s, #0 ; \t # compare the register with 0  " % (register))
                    codeprint("vmrs    APSR_nzcv, FPSCR ")
                    codeprint("bgt     .L%d" % (label_counter) )
                    codeprint(".ANTIL%d:" % label_counter)
                    codeprint("vmsr fpexc, r3")
                    codeprint("VMOV.F32 %s,%s  ; \t # put the value to 0 if less than 0" % (register,zero_register))
                    codeprint(".L%d: " % (label_counter))
                    codeprint("vmsr fpexc, r3")
                    label_counter += 1
                    
                if f_name == 'LINEAR':
                    #the function is Linear activation  function
                    0 # do nothing 
            else:
                # the variable is in memory
                address = 4 * allocator.memory_allocation_data[cursor_allocator].get_data()[tmp_name]
                bias_bytes = struct.pack('f', bias)
                bias_bytes_upper = struct.unpack('H',bias_bytes[:2])
                bias_bytes_lower = struct.unpack('H',bias_bytes[2:])
                
                #print("LOAD ADDRESS #%d INTO REGISTER $%s " %
                # (
                #     address,
                #     reg_1
                # )
                #)
                codeprint("FLDS %s,[r7, #%d] \t ; # load the content of the output neuron in buffer_register_2 " % (buffer_register_2, address))
                codeprint("MOVW r1,#%d \t ; # move the most significative bits in r3" % bias_bytes_upper[0])
                codeprint("MOVT r1,#%d \t ; # move the least significative bits in r3" % bias_bytes_lower[0])
                codeprint("VMOV.F32 %s,r1 \t ; # copy r3 in buffer_register_1" % (buffer_register_1))
                codeprint("VADD.F32 %s,%s,%s \t ; # add the bias" % (
                        buffer_register_2,
                        buffer_register_1,
                        buffer_register_2
                ))
                
                #print("ADDI $%s,$%s,#%f" %
                #      (
                #          reg_1,
                #          reg_1,
                #          bias
                #      ))
                if f_name == 'RELU':
                    #the function is RELU
                    #print("GTZ $%s TO $%s" %   # if greater than 0 save 1 in $0 else 0 in $0
                    #         (
                    #             reg_1,
                    #             reg_0
                    #         )
                    #     )
                    #print("MULT $%s, $%s, $%s" %
                    #         (
                    #             reg_1,
                    #             reg_0,
                    #             reg_1
                    #         )
                    #     )
                    codeprint("# RELU")
                    codeprint("vcmpe.f32  %s, #0 ; \t # compare the register with 0  " % (buffer_register_2))
                    codeprint("vmrs    APSR_nzcv, FPSCR ")
                    codeprint("bgt     .L%d" % (label_counter) )
                    codeprint(".ANTIL%d:" % label_counter)
                    codeprint("vmsr fpexc, r3")
                    codeprint("VMOV.F32 %s,%s  ; \t # put the value to 0 if less than 0" % (buffer_register_2,zero_register))
                    codeprint(".L%d: " % (label_counter))
                    codeprint("vmsr fpexc, r3")
                    label_counter += 1
                    
                if f_name == 'LINEAR':
                    #the function is Linear activation  function
                    0
                
                #print("STORE  $%s  INTO ADDRESS #%d" %
                # (
                #     reg_1,
                #     address
                # )
                #)
                codeprint("FSTS %s,[r7, #%d] " % (buffer_register_2, address))
    return ret, input_mask, output_mask

rete = Program([10,30,30,2], ["RELU","RELU","LINEAR"])
rete.random_init_weights()
codice, input_mask, output_mask = compiler(rete, [ 's' + str(i) for i in range(16)] , r7offset = 0x1000000)
print(codice)