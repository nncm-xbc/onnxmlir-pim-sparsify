"""Compiler from a sparsified MLP to ARMv7-A VFP assembly.

Pipeline:

1. :class:`Program` — semantic representation of the MLP (weights, biases,
   activations).
2. :func:`IR` — lower :class:`Program` to a tree-based intermediate
   representation built from :class:`TreeNode`. Zero weights are skipped
   to exploit sparsity.
3. :class:`Allocator` — assign every IR temporary to either a hard register
   or a memory slot. Memory placement is optimized via simulated annealing
   on a per-layer signal-distance matrix.
4. :func:`compiler` / :func:`executable` — emit ARMv7 VFP assembly using
   the allocation, plus a wrapper exposing the C ABI symbol
   ``network_inference``.

Key public API: :class:`Program`, :func:`IR`, :class:`Allocator`,
:func:`compiler`, :func:`executable`, :func:`net_to_torch`,
:func:`torch_to_onnx`.
"""

import logging
import os
import struct
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from backend.network import Network
from mlp.mlp import *

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

_log = logging.getLogger(__name__)

# Bytes per memory slot. Slots hold single-precision floats, so memory
# addresses are emitted as ``[r7, #(slot * FLOAT_BYTES)]``.
FLOAT_BYTES = 4

################################################################################################################################

class Program:
    """Semantic representation of a trained MLP.

    Stores per-layer weight matrices ``W``, bias vectors ``b``, the network
    topology (layer sizes), and a list of activation function names. The
    object is the input to :func:`IR` and downstream code generation.

    Constructed either from a folder of saved parameters (passing the path
    as ``topology``) or from an explicit topology with activations.
    """

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
        else:                     # empty network of a given topology
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
            print("Activation functions = %s" % str(self.activation_functions_list[i]))

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
        

################################################################################################################################
#                                                   INTERMEDIATE REPRESENTATION                                                #
################################################################################################################################


class TreeNode:
    """Node in the IR expression tree.

    Attributes:
        id: opcode/identifier of the node (e.g. ``"MOVE"``, ``"BINOP"``,
            or a numeric/string leaf value).
        sons: ordered list of child :class:`TreeNode` instances. Leaves
            (``ADD``, ``MUL``, ...) have an empty ``sons`` list.
    """
    def __init__(self, ID, *args):
        self.id   = ID
        self.sons = list(args)

    def print(self, level=0):
        print(("\t" * level) + str(self.id))
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
        ret = [self.id]
        for s in self.sons:
            ret += s.flatten()
        return ret
            
#                                            >>>INTERMEDIATE REPRESENTATION SYNTHESIS<<<
#                                    * Given a MultiLayer Perceptron represented through the class
#                                    * Program produces a Tree based intermediate representation
#                                    * which is used by the compiler to generate assembly code.

def IR(program, compile_time_data=True):
    """Lower a :class:`Program` to a flat list of :class:`TreeNode` IR statements.

    For each layer the IR contains:

    - ``COMMENT(START)`` / ``COMMENT(END)`` block markers (one block per
      matrix multiplication).
    - One ``MOVE(TEMP, CONST(0))`` to zero each output accumulator.
    - One ``MOVE(TEMP, BINOP(ADD, TEMP, BINOP(MUL, CONST(w), TEMP)))`` per
      non-zero weight ``w`` (sparsity-aware: zeros are skipped).
    - One fused ``MOVE(CALL(activation, TEMP, BINOP(ADD, TEMP, CONST(b))))``
      per output, applying bias and activation in a single statement.

    Args:
        program: source :class:`Program`.
        compile_time_data: kept for API compatibility; currently unused.

    Returns:
        list of :class:`TreeNode` instructions in execution order.
    """
    ret = list()

    for layer,t in enumerate(program.topology[1:]):
        IR_instruction = TreeNode("COMMENT", TreeNode("START"))
        ret.append(IR_instruction)

        f = program.activation_functions_list[layer]
        R_offset = program.topology[layer]
        for i in range(len(program.b[layer])):
            IR_instruction = TreeNode(
                    "MOVE",
                     TreeNode("TEMP" , TreeNode(i + R_offset) ),
                     TreeNode("CONST", TreeNode(0))
            )
            ret.append(IR_instruction)

        for i in range(len(program.W[layer])):
            for j in range(len(program.W[layer].T)):
                if program.W[layer][i,j] != 0:
                    # Skip zero weights to exploit sparsity
                    IR_instruction = TreeNode(
                        "MOVE",
                        TreeNode("TEMP",
                            TreeNode(i + R_offset)
                        ),
                        TreeNode("BINOP",
                            TreeNode("ADD"),
                            TreeNode("TEMP",
                                TreeNode(i + R_offset)
                            ),
                            TreeNode("BINOP",
                                TreeNode("MUL"),
                                TreeNode("CONST",
                                     TreeNode(program.W[layer][i,j])
                                 ),
                                TreeNode("TEMP",
                                 TreeNode(j)
                                 )
                            )
                        )
                    )
                    ret.append(IR_instruction)

        for i in range(len(program.b[layer])):
            # Bias add and activation are fused into a single MOVE+CALL,
            # so that the post-activation value is written without an
            # intermediate read/write on the temporary.
            IR_instruction = TreeNode("MOVE",
                                TreeNode("CALL",
                                    TreeNode(f),
                                    TreeNode("TEMP",
                                     TreeNode(i + R_offset)
                                     ),
                                    TreeNode("BINOP",
                                     TreeNode("ADD"),
                                     TreeNode("TEMP",
                                       TreeNode(i + R_offset)
                                      ),
                                     TreeNode("CONST",
                                      TreeNode(program.b[layer][i])
                                      )
                                     )
                                 )
                                )
            ret.append(IR_instruction)

        IR_instruction = TreeNode("COMMENT", TreeNode("END"))
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
    """Per-block usage histogram for IR temporaries.

    Used to drive register allocation: temporaries that appear most often
    in a matrix-multiplication block are preferred for register assignment.
    """

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


class RegisterAllocationData:
    """Mapping from IR temporaries to physical (or placeholder) register names.

    Placeholder registers are named ``"register_<i>"`` and are resolved to
    concrete VFP registers later by the inter-block flow analysis. Concrete
    registers carry their assembly name directly (e.g. ``"s4"``).

    By convention, temporary IDs less than ``prev_layer_size`` are inputs
    of the current block (outputs of the previous block); the rest are
    outputs of the current block.
    """

    def __init__(self):
        self.temp_reg_map = {}

    def get(self, temp_variable):
        return self.temp_reg_map[temp_variable]

    def insert(self, temp_variable, register):
        self.temp_reg_map[temp_variable] = register

    def get_data(self):
        return self.temp_reg_map

    def rename(self, old_reg_name, new_reg_name):
        for t in self.temp_reg_map:
            if self.temp_reg_map[t] == old_reg_name:
                self.temp_reg_map[t] = new_reg_name

    def get_unitialized_temps(self):
        return [s for s in self.temp_reg_map if self.temp_reg_map[s].startswith("register_")]

    def get_initialized_registers(self):
        return [
            self.temp_reg_map[s]
            for s in self.temp_reg_map
            if not self.temp_reg_map[s].startswith("register_")
        ]

    def get_variables_list(self):
        return list(self.temp_reg_map)

    def get_input_temps(self, prev_layer_size):
        all_vars = np.array(self.get_variables_list())
        return np.array([
            [var, self.temp_reg_map[var]] for var in all_vars if var < prev_layer_size
        ])

    def get_output_temps(self, prev_layer_size):
        all_vars = np.array(self.get_variables_list())
        return np.array([
            [var, self.temp_reg_map[var]] for var in all_vars if var >= prev_layer_size
        ])

    def print(self):
        for t in self.temp_reg_map:
            print(t, "\t", self.temp_reg_map[t])

    def contains(self, tmp_name):
        return tmp_name in self.temp_reg_map
        
class MemoryAllocationData:
    """Mapping from IR temporaries to memory slot indices.

    A slot index of ``-1`` denotes an unallocated temporary. Final slot
    indices are byte addresses scaled by ``4`` (single-precision floats)
    when emitted as ``[r7, #addr]`` operands.
    """

    def __init__(self):
        self.temp_mem_map = {}

    def insert(self, temp_variable, address):
        self.temp_mem_map[temp_variable] = address

    def batch_set(self, list_of_temps, list_of_addresses):
        for tmp_id, mem_addr in zip(list_of_temps, list_of_addresses):
            self.temp_mem_map[tmp_id] = mem_addr

    def get_data(self):
        return self.temp_mem_map

    def get(self, x):
        return self.temp_mem_map[x]

    def get_variables_list(self):
        return list(self.temp_mem_map)

    def get_input_temps(self, prev_layer_size):
        all_vars = np.array(self.get_variables_list())
        return all_vars[all_vars < prev_layer_size]

    def get_output_temps(self, prev_layer_size):
        all_vars = np.array(self.get_variables_list())
        return all_vars[all_vars >= prev_layer_size]

    def print(self):
        for t in self.temp_mem_map:
            print(t, "\t", self.temp_mem_map[t])


################################################################################################################################
#                                          REGISTER AND MEMORY ALLOCATION STRATEGY                                             #
################################################################################################################################


class BlockSignals:
    """Per-temporary boolean usage signal across a matrix-multiplication block.

    For each memory-resident temporary, stores a list whose ``i``-th entry
    is ``1`` iff the temporary is read or written by the ``i``-th IR
    statement of the block. The pairwise distance between these signals
    drives memory-slot placement.
    """

    def __init__(self, memory_allocation_object):
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
        
################################################################################################################################
#                                                  INTER-BLOCK DATA FLOWS                                                     #
################################################################################################################################


class MemoryToRegisterFlow:
    """Move from a memory slot into a VFP register at a layer boundary."""

    def __init__(self, mem_address, register):
        self.mem_address = mem_address
        self.register    = register
    def print(self):
        print("M2R flow\t%s\t->\t%s" % (self.mem_address,self.register))
        
        
class RegisterToMemoryFlow:
    """Move from a VFP register into a memory slot at a layer boundary."""

    def __init__(self, register, mem_address):
        self.mem_address = mem_address
        self.register    = register
    def print(self):
        print("R2M flow\t%s\t->\t%s" % (self.register,self.mem_address))

class RegisterRenameFlow:
    """Resolve a placeholder register name to a concrete VFP register."""

    def __init__(self, register_placeholder, register_name):
        self.register_placeholder = register_placeholder
        self.register_name        = register_name
    def print(self):
        print("R2R flow\t%s\t->\t%s" % (self.register_placeholder,self.register_name))


class InterfaceCommunication:
    """Set of data flows between two consecutive matrix-multiplication blocks.

    Holds three lists, one per flow kind: register-to-memory,
    memory-to-register, and register-rename. Consumed by
    :func:`interfaces_manager` to emit the assembly that materialises the
    layer boundary.
    """

    def __init__(self):
        self.reg2mem_flows = list()
        self.mem2reg_flows = list()
        self.reg2reg_flows = list()
    def insert(self,flow):
        if type(flow).__name__ == "RegisterToMemoryFlow":
            self.reg2mem_flows.append(flow)
        else:
            if type(flow).__name__ == "MemoryToRegisterFlow":
                self.mem2reg_flows.append(flow)
            else:
                if type(flow).__name__ == "RegisterRenameFlow":
                    self.reg2reg_flows.append(flow)
                    
    def get_reg2mem_flows(self):
        return self.reg2mem_flows
    def get_mem2reg_flows(self):
        return self.mem2reg_flows
    def get_reg2reg_flows(self):
        return self.reg2reg_flows
    
    def print(self):
        reg2mem = self.get_mem2reg_flows()
        mem2reg = self.get_reg2mem_flows()
        reg2reg = self.get_reg2reg_flows()
        for rm in reg2mem:
            rm.print()
        for mr in mem2reg:
            mr.print()
        for rr in reg2reg:
            rr.print()

################################################################################################################################
#                                                        ALLOCATOR CLASS                                                       #
################################################################################################################################

class Allocator:
    """Per-block register and memory allocation for the IR.

    For every matrix-multiplication block in the IR, decides whether each
    temporary lives in a register or a memory slot, and produces the set
    of inter-block data flows (:class:`InterfaceCommunication`) needed to
    move values between layers.

    The pipeline is:

    1. :meth:`most_used_temps` — rank temporaries per block by usage.
    2. :meth:`register_allocation_and_memory_alloc_init` — assign the most
       used temporaries to registers; the rest are flagged for memory.
    3. :meth:`compute_signals` / :meth:`compute_signals_distance_matrix` —
       build a per-block matrix of pairwise temporal distances between
       memory-resident temporaries.
    4. :meth:`memory_allocation` — solve memory placement via simulated
       annealing (:meth:`anneal`) and density heuristics
       (:meth:`density_optimizer_memory_subset_for_output`), and record the
       inter-block flows.
    """

    def __init__(self, ir, program, register_names):
        self.register_allocation_data = []  # one RegisterAllocationData per matmul block
        self.memory_allocation_data   = []  # one MemoryAllocationData per matmul block
        self.interfaces               = {}
        self.register_names           = register_names
        self.program                  = program

        temp_statistics = self.most_used_temps(ir)
        self.register_allocation_and_memory_alloc_init(temp_statistics, register_names)
        self.memory_allocation(
            self.compute_signals_distance_matrix(self.compute_signals(ir))
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
                ret[temp] = ("mem", FLOAT_BYTES * int(self.memory_allocation_data[0].get_data()[temp]))
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
                ret[temp - self.program.topology[-2]] = ("mem", FLOAT_BYTES * int(self.memory_allocation_data[-1].get_data()[temp]))
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
            arr = stat.vectorize()
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

        return 0
    
    ########################################################################################
    # Memory allocation optimization
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
                    distance = 0.5 * (np.mean([ np.min(np.abs(s_1 - v_b)) for s_1 in v_a]) + np.mean([ np.min(np.abs(s_2 - v_a)) for s_2 in v_b]))
                    D[i,j] = distance
            Ds.append(D)
            mappings.append(mapping)
            inverse_mappings.append(inverse_mapping)
        return Ds, mappings, inverse_mappings
    
    def anneal(self,
               unconstrained_temps,     # temporary unconstrained
               addresses,              # list of available memory addresses
               Ds,                     # distance matrix
               inverse_mapping):       # mapping between rows of the distance matrix and temps
        # initialize a random association
        association = []
        for u,a in zip(unconstrained_temps,addresses):
            association.append([u,a])
        association = np.array(association)
        
        # nothing to optimize actually...
        if(len(unconstrained_temps) == 1):
            return association
        
        decay = 1e-2 #decay = 1e-3
        T = 10.
        T_end = 1e-3
        
        inverse_mapping_unconstrained = {}
        for i,u in enumerate(unconstrained_temps):
            inverse_mapping_unconstrained[u] = i
        
        Ds_unconstrained = Ds[
            [ inverse_mapping[u] for u in unconstrained_temps], :
        ][ :, [ inverse_mapping[u] for u in unconstrained_temps]]
        
        
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

            move = np.arange(len(association)).astype(int)
            # swap two random
            a = np.random.choice(len(move))
            b = np.random.choice(len(move))
            tmp = move[a]
            move[a] = move[b]
            move[b] = tmp
            new_association      = association.copy()
            new_association[:,1] = association[:,1][move]
            
            dE = cost(new_association) - cost(association)
            
            if dE <= 0.:
                association = new_association
            else:
                if( np.random.uniform() > np.exp( - T)):
                    association = new_association
            
            if anneal_count % int(anneal_iterations / 10) == 0:
                _log.debug("anneal cost = %.2f", cost(association))
                
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
        memory_mask = np.arange(len(memory))[ [ not(x in m_i_on_constraint ) for x in np.arange(len(memory))] ]

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
            # Among configurations with maximal density, prefer the one
            # where the unconstrained block sits closest to the constrained block.
            available_configurations = rows[densities == densities.max()]
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

            return memory[configuration > 0.]
        

####################################################################################################################
#####                  #############################################################################################
#####   memory alloc   #############################################################################################
#####                  #############################################################################################
####################################################################################################################

    def memory_allocation(self, DS_MAPPINGS):
        # Allocate a memory pool sized to the largest layer's memory-resident temps.
        memory = np.arange(
            np.max(
                [
                       np.max(len(m.get_variables_list()))
                    for m in self.memory_allocation_data
                ]
            )
        )

        # Layer 0: every memory-resident temp is unconstrained.
        _log.info("Matrix Multiplication %d", 0)
        T_mem_1       = self.memory_allocation_data[0].get_variables_list()
        mem_addresses = np.arange(0, len(T_mem_1))
        Ds_1                      = DS_MAPPINGS[0][0]
        mapping_1                 = DS_MAPPINGS[1][0]
        inverse_mapping_1         = DS_MAPPINGS[2][0]
        # optimize the                 
        if len(T_mem_1) > 0:
            _log.info("Optimizing memory allocation of unconstrained temps...")
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
        

        for i in range(1, len(DS_MAPPINGS[0])):
            _log.info("Matrix Multiplication %d", i)
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
            free_registers = set()
            for r in self.register_names:
                free_registers.add(r)


            # if some output stays in registers when it becomes input we dont want to move it
            for t in T_in_curr.intersection(T_reg_curr):
                if single_phi_inv(t) in ( T_out_prev.intersection(T_reg_prev)):
                    r  = self.register_allocation_data[i - 1].get(
                                                               single_phi_inv(t)
                                                           )
                    self.register_allocation_data[i].insert(t, 
                                                            r
                                                           )
                    free_registers.remove(r)
                    
            # from memory to registers
            for t in T_in_curr.intersection(T_reg_curr):
                if single_phi_inv(t) in T_out_prev.intersection(T_mem_prev):
                    r = list(free_registers)[0]
                    self.register_allocation_data[i].insert(t, r)
                    free_registers.remove(r)
                    self.interfaces[i-1,i].insert(
                        MemoryToRegisterFlow(
                            self.memory_allocation_data[i - 1].get(single_phi_inv(t)),
                            r
                        )
                    )
                    
            # outputs that go into registers
            for t in T_out_curr.intersection(T_reg_curr):
                r = list(free_registers)[0]
                self.register_allocation_data[i].insert(t, r)
                free_registers.remove(r)
            
            
            # Memory
            
            unconstrained_temps = T_mem_curr.copy()
            constrained_temps   = set()
            
            for t in T_in_curr.intersection(T_mem_curr):
                if single_phi_inv(t) in T_out_prev.intersection(T_mem_prev):
                    self.memory_allocation_data[i].insert( 
                        t , 
                        self.memory_allocation_data[i - 1].get(single_phi_inv(t))
                    )       
                    unconstrained_temps.remove(t)
                    constrained_temps.add(t)
                    
            if len(unconstrained_temps) > 0:
                _log.info("Optimizing memory allocation of unconstrained temps...")
                
                memory_address_image_of_mi = self.density_optimizer_memory_subset_for_output(
                    memory,                           # memory object
                    constrained_temps,                 # set temporaries that are constrained
                    self.memory_allocation_data[i],   # mapping m_i
                    len(unconstrained_temps)           # length of the unconstrained temps to place  
                ) # returns a list of addresses that suit the unconstrained variables

                
                Ds              = DS_MAPPINGS[0][i]
                mapping         = DS_MAPPINGS[1][i]
                inverse_mapping = DS_MAPPINGS[2][i]
                
            
                temp_addr_mapping = self.anneal(    unconstrained_temps, 
                                                    memory_address_image_of_mi,
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


def interfaces_manager(interface, buffer_register_1):
    """Emit assembly that materialises a layer-boundary :class:`InterfaceCommunication`.

    Builds a directed dependency graph over the reg→mem and mem→reg flows
    and traverses it. Acyclic chains ("threads") are emitted as a sequence
    of ``FSTS``/``FLDS``. Cycles ("loops") are broken by spilling the head
    register to ``buffer_register_1`` before walking the cycle backwards.

    Args:
        interface: the :class:`InterfaceCommunication` between two
            consecutive blocks.
        buffer_register_1: scratch VFP register used to break cycles.

    Returns:
        Newline-separated ARMv7 assembly as a string (no trailing newline).
    """
    out_lines = []

    def codeprint_function(line):
        out_lines.append(line)
    
    
    G = nx.DiGraph()
    selectable_registers = set()
    for r2m in interface.get_reg2mem_flows():
        G.add_edge("reg_" + str(r2m.register), "mem_" + str(r2m.mem_address))
        G.nodes["mem_" + str(r2m.mem_address)]['in'] = "reg_" + str(r2m.register)
        G.nodes["reg_" + str(r2m.register)  ]['out'] = "mem_" + str(r2m.mem_address) 
        
        selectable_registers.add("reg_" + str(r2m.register))
        
    for m2r in interface.get_mem2reg_flows():
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
                    address  = FLOAT_BYTES * int(cursor_id.split("_")[1])
                    # // dest is memory
                    codeprint_function("FSTS %s,[r7,#%s]" %  (register, address))
                # if source is memory
                else:
                    # // dest is register
                    register  = cursor_id.split("_")[1]
                    address = FLOAT_BYTES * int(cursor_father_id.split("_")[1])
                    codeprint_function("FLDS %s,[r7,#%s]" %  (register, address))
    #           cursor = cursor.prev
                cursor_id = cursor.get("in")
                cursor    = G.nodes[cursor_id]
        else:
    #       // this is a loop
    #       codeprint [ copy head into BUFFER_REGISTER_1] // head_copy
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
                    address  = FLOAT_BYTES * int(cursor_id.split("_")[1])
                    # // dest is memory
                    codeprint_function("FSTS %s,[r7,#%s]" %  (register, address))
                # if source is memory
                else:
                    # // dest is register
                    register  = cursor_id.split("_")[1]
                    address = FLOAT_BYTES * int(cursor_father_id.split("_")[1])
                    codeprint_function("FLDS %s,[r7,#%s]" %  (register, address))
    #           cursor = cursor.prev
                cursor_id = cursor.get("in")
                cursor    = G.nodes[cursor_id]
    #       codeprint [ copy BUFFER_REGISTER_1 into cursor] // head_copy
            register = cursor_id.split("_")[1]
            codeprint_function("VMOV.F32 %s,%s" % (register, buffer_register_1))    
    return "\n".join(out_lines)
 

            
################################################################################################################################
#                                                      COMPILER FUNCTION                                                       #
################################################################################################################################


def compiler(network, registers, sparsify=False, r7offset=0):
    """Emit ARMv7-A VFP assembly for a :class:`Program` network.

    Lowers ``network`` to IR, allocates registers and memory, then walks
    the IR statement-by-statement and prints the corresponding assembly,
    inserting per-layer interface code.

    Args:
        network: :class:`Program` to compile.
        registers: VFP register names. ``registers[0]`` is reserved as the
            zero register, ``registers[1..2]`` as scratch buffers, and the
            rest are available for allocation.
        sparsify: kept for API compatibility; sparsity is already exploited
            during IR generation.
        r7offset: byte offset loaded into ``r7`` at function entry. Memory
            slots are addressed as ``[r7, #<addr>]``.

    Returns:
        Tuple ``(asm_code, input_mask, output_mask)``. ``asm_code`` is the
        body of the ``network_inference`` function as a newline-separated
        string. The masks describe where the network's input/output entries
        live (register or memory) so that :func:`executable` can emit the
        correct prologue/epilogue.
    """
    out_lines = []

    def codeprint(line):
        out_lines.append(line)

    reg_0 = registers[0]
    reg_1 = registers[1]

    intermediate_representation = IR(network)

    allocator   = Allocator(intermediate_representation, network, registers[3:])
    input_mask  = allocator.get_input_mapping()    # dictionary that maps inputs entries to registers/memory units
    output_mask = allocator.get_output_mapping()   # dictionary that maps outputs entries to register/memory units
    
    #Save IR
    with open(__location__ +"/IR", "w") as f:
        for ir_statement in intermediate_representation:
            f.write(str(ir_statement) + "\n")
    
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
    
    cursor_allocator = 0
    label_counter = 0

    for ir_statement in intermediate_representation:
        flattened = ir_statement.flatten()
        if flattened[0] == "COMMENT":
            if flattened[1] == "END":
                codeprint("######################################## INTERFACE")
                if cursor_allocator < len(allocator.interfaces):
                    codeprint("# Reading interface [%d,%d]" % (cursor_allocator, cursor_allocator + 1))
                    interface = allocator.interfaces[cursor_allocator, cursor_allocator + 1]
                    codeprint( interfaces_manager(interface, buffer_register_1) )
                cursor_allocator += 1
        
        if(flattened[0] == "MOVE" 
           and 
           flattened[3] == "CONST"
          ): # set the temporary variable
            tmp_name =  flattened[2]
            val       = flattened[4]
            # we have to understand if tmp_name is a register or not
            if allocator.register_allocation_data[cursor_allocator].contains(tmp_name):
                # variable is in registers
                register = allocator.register_allocation_data[cursor_allocator].get_data()[tmp_name]
                codeprint("VMOV.F32 %s, %s\t ; # set the register to 0 " % (register,zero_register))
            else:
                # variable is in memory
                address = FLOAT_BYTES * allocator.memory_allocation_data[cursor_allocator].get_data()[tmp_name]
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
            
            
            if allocator.register_allocation_data[cursor_allocator].contains(neuron_source):
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
            else:
                # variable is in memory
                address = FLOAT_BYTES * allocator.memory_allocation_data[cursor_allocator].get_data()[neuron_source]
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

            # buffer_register_2 now contains weight * input
            if allocator.register_allocation_data[cursor_allocator].contains(neuron_dest):
                # dest += weight * input
                codeprint("VADD.F32 %s,%s,%s \t ; # " %(
                    allocator.register_allocation_data[cursor_allocator].get_data()[neuron_dest],
                    buffer_register_2,
                    allocator.register_allocation_data[cursor_allocator].get_data()[neuron_dest]))
            else:
                # variable is in memory
                address = FLOAT_BYTES * allocator.memory_allocation_data[cursor_allocator].get_data()[neuron_dest]
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
            
            if allocator.register_allocation_data[cursor_allocator].contains(tmp_name):
                # the variable is in a register
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
                    pass
            else:
                # the variable is in memory
                address = FLOAT_BYTES * allocator.memory_allocation_data[cursor_allocator].get_data()[tmp_name]
                bias_bytes = struct.pack('f', bias)
                bias_bytes_upper = struct.unpack('H',bias_bytes[:2])
                bias_bytes_lower = struct.unpack('H',bias_bytes[2:])

                codeprint("FLDS %s,[r7, #%d] \t ; # load the content of the output neuron in buffer_register_2 " % (buffer_register_2, address))
                codeprint("MOVW r1,#%d \t ; # move the most significative bits in r3" % bias_bytes_upper[0])
                codeprint("MOVT r1,#%d \t ; # move the least significative bits in r3" % bias_bytes_lower[0])
                codeprint("VMOV.F32 %s,r1 \t ; # copy r3 in buffer_register_1" % (buffer_register_1))
                codeprint("VADD.F32 %s,%s,%s \t ; # add the bias" % (
                        buffer_register_2,
                        buffer_register_1,
                        buffer_register_2
                ))

                if f_name == 'RELU':
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
                    pass

                codeprint("FSTS %s,[r7, #%d] " % (buffer_register_2, address))
    return "\n".join(out_lines) + "\n", input_mask, output_mask




def executable(network, registers, sparsify=False, r7offset=0x1000000):
    """Wrap :func:`compiler`'s output in a self-contained ARMv7 function.

    Adds a ``network_inference`` symbol with the C ABI ``(float* input,
    float* output)``, the prologue/epilogue saving callee-saved registers
    and VFP registers, and per-input/output ``vldr``/``vstr`` glue based
    on the masks returned by :func:`compiler`.

    Returns:
        Tuple ``(asm_code, executable_code, input_mask, output_mask)``.
    """
    asm_code, input_mask, output_mask = compiler(network, registers, sparsify, r7offset)

    formatted_asm = asm_code
    formatted_asm = formatted_asm.replace("VMUL.F32 s", "vmul.f32 s")
    formatted_asm = formatted_asm.replace("VADD.F32 s", "vadd.f32 s")
    formatted_asm = formatted_asm.replace("VSUB.F32 s", "vsub.f32 s")
    formatted_asm = formatted_asm.replace("VMOV.F32 s", "vmov.f32 s")
    formatted_asm = formatted_asm.replace(",s", ", s")
    formatted_asm = formatted_asm.replace(",r", ", r")
    formatted_asm = formatted_asm.replace("FLDS s", "flds s")
    formatted_asm = formatted_asm.replace("FSTS s", "fsts s")
    formatted_asm = formatted_asm.replace(",[r", ", [r")

    #function wrapper
    wrapper = []
    wrapper.append(".arch armv7-a")
    wrapper.append(".fpu vfp")
    wrapper.append(".section .text")
    wrapper.append(".align 4")
    wrapper.append(".global network_inference")
    wrapper.append(".type network_inference, %function")
    wrapper.append("")
    wrapper.append("@ network_inference(float* input, float* output)")
    wrapper.append("@ r0: input pointer")
    wrapper.append("@ r1: output pointer")
    wrapper.append("")
    wrapper.append("network_inference:")
    wrapper.append("    push {r4-r11, lr}       @ Save registers")
    wrapper.append("    vpush {s16-s31}         @ Save VFP registers")

    #load inputs to registers/memory
    wrapper.append("")
    wrapper.append("    @ Load inputs")
    for idx, (loc_type, location) in input_mask.items():
        if loc_type == "reg":
            wrapper.append(f"    vldr {location}, [r0, #{idx*FLOAT_BYTES}]")
        else:  # memory location
            wrapper.append(f"    vldr s0, [r0, #{idx*FLOAT_BYTES}]")
            wrapper.append(f"    vstr s0, [r7, #{location}]")

    wrapper.append("")
    wrapper.append("    @ Network computation")
    wrapper.append(formatted_asm)

    #store outputs
    wrapper.append("")
    wrapper.append("    @ Store outputs")
    for idx, (loc_type, location) in output_mask.items():
        if loc_type == "reg":
            wrapper.append(f"    vstr {location}, [r1, #{idx*FLOAT_BYTES}]")
        else:  # memory location
            wrapper.append(f"    vldr s0, [r7, #{location}]")
            wrapper.append(f"    vstr s0, [r1, #{idx*FLOAT_BYTES}]")
    
    wrapper.append("")
    wrapper.append("    vpop {s16-s31}          @ Restore VFP registers")
    wrapper.append("    pop {r4-r11, pc}        @ Restore registers and return")

    executable_code = "\n".join(wrapper)
    return asm_code, executable_code, input_mask, output_mask

def net_to_torch(network):
    """Wrap a :class:`Program` in a :class:`backend.network.Network` ``nn.Module``."""
    return Network(network)


def torch_to_onnx(network, output_file):
    """Export ``network`` to ``<output_file>.onnx`` via ``torch.onnx``."""
    example_input = torch.randn(1, network.topology[0])
    net_onnx = torch.onnx.export(network, example_input, dynamo=True)
    net_onnx.optimize()
    net_onnx.save(output_file + ".onnx")

if __name__ == "__main__":

    args              = sys.argv
    parameters_folder = os.path.abspath(args[1])
    output_file       = os.path.abspath(args[2])

    network = Program(parameters_folder)
    torchnet= net_to_torch(network)

    input_tensor = torch.randn(1, 196)

    print("Output from PyTorch model:")
    output_tensor = torchnet(input_tensor)
    print(output_tensor)
    torch.save(torchnet, output_file + ".pt")

    torch_to_onnx(torchnet, output_file)

    print("Output from OG network:")
    output = network.run(input_tensor.numpy().squeeze(0))
    print(output)

    asm_code, executable_code, input_mask, output_mask = executable(network, ['s' + str(i) for i in range(16)], r7offset=0x1000000)

    with open(output_file + "_exe", "w") as out:
        out.write(executable_code)

    with open(output_file, "w") as out:
        out.write(asm_code)
